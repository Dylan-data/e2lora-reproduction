"""Evaluate a suite checkpoint without taking an optimizer step."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'src'))
from common import file_sha256, load_local_model, output_path, read_json, resolve, seed_everything, write_json
from suite_plan import validate


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run-dir',required=True)
    p.add_argument('--epoch',required=True,type=int)
    args=p.parse_args();directory=output_path(args.run_dir)
    run=read_json(directory/'run_config.json');cfg=validate(run['config'])
    if args.epoch<1 or args.epoch>cfg['training']['epochs']:p.error('epoch out of range')
    evaluation=directory/f'evaluation-epoch{args.epoch}.json'
    if evaluation.exists():print(json.dumps(read_json(evaluation),indent=2));return
    from train import require_gpu_allocation,load_complete_adapter_state
    require_gpu_allocation()
    import torch
    from e2lora.adapters.lora_layers import apply_plan
    from e2lora.adapters.partial_rank import count_adapters
    from team_rank_check import attach
    import suite_train
    for name,expected in run['source_hashes'].items():
        if file_sha256(resolve(name))!=expected:raise ValueError('Training source changed: '+name)
    state=torch.load(directory/'checkpoint.pt',map_location='cpu',weights_only=False)
    if state['identity']!=run['identity'] or state['plan']!=run['plan']:
        raise ValueError('Checkpoint identity/plan differs')
    import math
    metadata=read_json(directory/'data-manifest.json')
    per_epoch=math.ceil(metadata['train_count']/cfg['training']['global_batch_size'])
    if state['step']!=args.epoch*per_epoch:
        raise ValueError('This checkpoint is not from the requested epoch; historical states are not fabricated')
    seed_everything(run['seed']);model,tokenizer=load_local_model(cfg,'cuda:0')
    if run['method']!='lora' and cfg['suite']['adapter_policy']=='team_partial_fraction':
        attach(model,state['plan'],cfg)
    else:
        apply_plan(model,state['plan'],alpha=cfg['adapter']['alpha'])
        for group in state['plan']:
            layer=model.get_submodule(group['layers'][0]);layer.down.float();layer.up.float()
    if count_adapters(model,state['plan'])['trainable_adapter_parameters']!=run['parameters']['trainable_adapter_parameters']:
        raise ValueError('Adapter count differs from training')
    load_complete_adapter_state(model,state['adapter'])
    suite_train.STOP=False
    import signal
    signal.signal(signal.SIGTERM,suite_train.stop_requested)
    if hasattr(signal,'SIGUSR1'):signal.signal(signal.SIGUSR1,suite_train.stop_requested)
    result=suite_train.evaluate(model,tokenizer,cfg,directory,run['phase']=='smoke',args.epoch)
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()
