"""Export task gradients, RMI and proxy entropy for Figure 1(b)."""
import csv
import json
from pathlib import Path
import random
from time import perf_counter
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from huggingface_hub import hf_hub_download
from e2lora.adapters.adaptive import collect_gradients, build_plan

root = Path(__file__).resolve().parents[2]
cfg = json.loads((root/"references/text-generation-setup.json").read_text())
out = root/"outputs/dylan/math-similarity-details"
out.mkdir(parents=True, exist_ok=True)
torch.manual_seed(42)
torch.set_num_threads(4)
model_cfg = cfg["model"]
source = {"repo": "meta-math/MetaMathQA", "revision": "aa4f34d3d2d3231299b5b03d9b3e5a20da45aa18"}
data_path = hf_hub_download(source["repo"], "MetaMathQA-395K.json", repo_type="dataset", revision=source["revision"])
rows = json.loads(Path(data_path).read_text())
ids = random.Random(42).sample(range(len(rows)), 64)
tokenizer = AutoTokenizer.from_pretrained(model_cfg["repo"], revision=model_cfg["revision"], local_files_only=True)
model = AutoModelForCausalLM.from_pretrained(model_cfg["repo"], revision=model_cfg["revision"], local_files_only=True, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda().eval()
model.config.use_cache = False
model.requires_grad_(False)
assert model.config.num_hidden_layers == 32
families = {p:[f"model.layers.{i}.self_attn.{p}_proj" for i in range(32)] for p in ["q","v"]}
manifest = []
start = perf_counter()
def batches():
    for step,index in enumerate(ids):
        row = rows[index]
        prompt = row["query"] + "\n" + row["response"]
        batch = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        labels = batch["input_ids"].clone()
        manifest.append(dict(row_index=index,source_id=row.get("id"),tokens=int(labels.numel())))
        yield {**{k:v.cuda() for k,v in batch.items()},"labels":labels.cuda()}
        print(f"gradient batch {step+1}/64 elapsed {perf_counter()-start:.1f}s",flush=True)
def loss_for_batch(model,batch):
    return model(**batch).loss, int(batch["labels"][:,1:].ne(-100).sum())
gradients = collect_gradients(model,[n for names in families.values() for n in names],batches(),loss_for_batch)
plan,diagnostics = build_plan(gradients,families,base_rank=8,rank_policy="paper")
records=[]
for projection,stats in diagnostics.items():
    for i,entropy in enumerate(stats["entropy"]):
        interval=next((a,b) for a,b in stats["intervals"] if a<=i<b)
        records.append(["entropy",projection,i,"",entropy,interval[0],interval[1]-1])
        for j,value in enumerate(stats["rmi"][i]):
            records.append(["RMI",projection,i,j,value,interval[0],interval[1]-1])
with (out/"statistics.csv").open("w",newline="") as f:
    writer=csv.writer(f)
    writer.writerow(["type","projection","layer_i","layer_j","value","group_start","group_end"])
    writer.writerows(records)
result=dict(model=model_cfg,dataset=source,seed=42,sample_ids=ids,samples=manifest,gradient_batches=64,micro_batch=1,max_sequence_length=512,precision="bf16",prompt="query newline response",loss_tokens="all nonpadding shifted tokens",gradient_averaging="weighted by effective prediction token count",optimizer_updates=0,projections=["q","v"],diagnostics=diagnostics,plan=plan,elapsed_seconds=perf_counter()-start)
(out/"result.json").write_text(json.dumps(result,indent=2))
torch.save(gradients,out/"average_gradients.pt")
print("COMPLETE",result["elapsed_seconds"],flush=True)
# Alternative metrics use the same averaged tensors and the same layer ordering.
metric_rows=[]
all_metrics={}
for projection,names in families.items():
    vectors=[gradients[n].flatten().double() for n in names]
    low=min(v.min().item() for v in vectors)
    high=max(v.max().item() for v in vectors)
    histograms=[]
    for v in vectors:
        h=torch.histc(v.float(),bins=128,min=low,max=high).double()+1e-12
        histograms.append(h/h.sum())
    matrices={name:[[0.0]*32 for _ in range(32)] for name in ["cosine","L1","L2","KL"]}
    for i,a in enumerate(vectors):
        for j,b in enumerate(vectors):
            values={
                "cosine":float(torch.dot(a,b)/(a.norm()*b.norm()).clamp_min(1e-30)),
                "L1":float((a-b).abs().sum()),
                "L2":float((a-b).norm()),
                "KL":float((histograms[i]*(histograms[i].log()-histograms[j].log())).sum()),
            }
            for metric,value in values.items():
                matrices[metric][i][j]=value
    matrices["RMI"]=diagnostics[projection]["rmi"]
    all_metrics[projection]=matrices
    for metric,matrix in matrices.items():
        for i,row in enumerate(matrix):
            for j,value in enumerate(row):metric_rows.append([metric,projection,i,j,value])
with (out/"similarities.csv").open("w",newline="") as f:
    writer=csv.writer(f);writer.writerow(["metric","projection","layer_i","layer_j","value"]);writer.writerows(metric_rows)
result["similarities"]=all_metrics
result["metric_definitions"]={"cosine":"dot/(L2 norm product)","L1":"sum absolute differences","L2":"Euclidean norm","KL":"directed KL between 128-bin gradient histograms; common per-projection range; epsilon 1e-12; no plotting transform"}
(out/"result.json").write_text(json.dumps(result,indent=2))
print("ALL METRICS COMPLETE",flush=True)
