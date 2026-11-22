"""Record 32-layer Q/K/V/O gradient entropy during AdaLomo full tuning."""
import csv
import json
import math
from pathlib import Path
import random
from time import perf_counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from lomo_optim import AdaLomo

root=Path(__file__).resolve().parents[2]
cfg=json.loads((root/"references/text-generation-setup.json").read_text())
out=root/"outputs/dylan/training-entropy-details"
out.mkdir(parents=True,exist_ok=True)
torch.manual_seed(42)
torch.set_num_threads(4)
source={"repo":"meta-math/MetaMathQA","revision":"aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"}
path=hf_hub_download(source["repo"],"MetaMathQA-395K.json",repo_type="dataset",revision=source["revision"])
rows=json.loads(Path(path).read_text())
ids=random.Random(42).sample(range(len(rows)),1024)
tokenizer=AutoTokenizer.from_pretrained(cfg["model"]["repo"],revision=cfg["model"]["revision"],local_files_only=True)
model=AutoModelForCausalLM.from_pretrained(cfg["model"]["repo"],revision=cfg["model"]["revision"],local_files_only=True,torch_dtype=torch.bfloat16,attn_implementation="sdpa").cuda()
model.config.use_cache=False
model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant":False})
model.train()
step=0
recorded={}
def capture(name):
    def hook(grad):
        if step%5==0:
            var=grad.detach().float().var(correction=0).clamp_min(1e-24)
            recorded[name]=float(.5*var.log()+.5*math.log(2*math.pi*math.e))
        return grad
    return hook
for layer in range(32):
    for projection in ["q","k","v","o"]:
        name=f"model.layers.{layer}.self_attn.{projection}_proj"
        model.get_submodule(name).weight.register_hook(capture(name))
optimizer=AdaLomo(model,lr=1e-4,loss_scale=1)
manifest=dict(model=cfg["model"],dataset=source,seed=42,sample_ids=ids,steps=1024,micro_batch=1,max_sequence_length=512,prompt="query newline response",loss_tokens="all shifted tokens",optimizer="AdaLomo",learning_rate=1e-4,record_every=5,precision="bf16",record_timing="gradient before update at optimizer step; first recorded step is 5")
(out/"manifest.json").write_text(json.dumps(manifest,indent=2))
start=perf_counter()
with (out/"entropy.csv").open("w",newline="") as f:
    writer=csv.writer(f);writer.writerow(["seed","optimizer_step","layer","projection","proxy_entropy"])
    for step,index in enumerate(ids,1):
        row=rows[index]
        batch=tokenizer(row["query"]+"\n"+row["response"],return_tensors="pt",truncation=True,max_length=512)
        batch={k:v.cuda() for k,v in batch.items()}
        labels=batch["input_ids"].clone()
        recorded.clear()
        loss=model(**batch,labels=labels).loss
        if not torch.isfinite(loss):raise RuntimeError("Nonfinite training loss")
        optimizer.fused_backward(loss,lr=1e-4)
        if step%5==0:
            assert len(recorded)==128,(step,len(recorded))
            for name,value in sorted(recorded.items()):
                layer=int(name.split(".")[2]);projection=name.split(".")[-1][0]
                writer.writerow([42,step,layer,projection,value])
            f.flush()
            print(f"step {step}/1024 loss {loss.item():.4f} elapsed {perf_counter()-start:.1f}s",flush=True)
        (out/"progress.json").write_text(json.dumps(dict(step=step,loss=loss.item(),elapsed_seconds=perf_counter()-start)))
print("COMPLETE",flush=True)
