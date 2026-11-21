from copy import deepcopy
import json
from pathlib import Path
import zipfile

import pytest
import torch
from torch import nn

import common
from e2lora.adapters.partial_rank import active_rank_plan, apply_partial_plan, adapter_state, count_adapters
from e2lora.adapters.lora_layers import apply_plan
from suite_plan import METHODS, active_budgets, family_plan, selection, specifications, validate
from run_ablations import command, smoke_ok, compile_suite
from summarize import collect, aggregate, report
from export_code import export
from watch_suite import retryable


@pytest.fixture
def suite():return common.read_json('experiments/chloe/configs/suite-math.json')


def test_matrix_deduplicates_tables_and_uses_only_explicit_seeds(suite):
    setup=common.read_json('references/ablation-setup.json')
    runs=specifications(suite,setup)
    assert {r['run']['seed'] for r in runs}=={42}
    keys=[(r['run']['method'],r['run']['rank'],r['run']['sampling_steps'],r['run']['seed']) for r in runs]
    assert len(keys)==len(set(keys))
    full=next(r for r in runs if r['run']['method']=='e2lora' and r['run']['rank']==8 and r['run']['sampling_steps']==64)
    assert set(full['run']['studies'])=={'table4','table5','figure5_rank','figure5_steps'}
    assert {r['run']['rank'] for r in runs if 'figure5_rank' in r['run']['studies']}=={4,8,16,32,64}
    assert {r['run']['sampling_steps'] for r in runs if 'figure5_steps' in r['run']['studies']}=={8,16,32,64,128,256,512}
    suite['seeds']=[17,23]
    assert len(specifications(suite,setup))==2*len(runs)


def test_nulls_and_unsupported_effective_settings_are_not_silently_ignored(suite):
    for path in [('suite','rank_sweep_fixed_steps'),('suite','step_sweep_fixed_rank'),('model','revision')]:
        cfg=deepcopy(suite);cfg[path[0]][path[1]]=None
        with pytest.raises(ValueError,match='null'):validate(cfg)
    for key,value in [('micro_batch_size',2),('gradient_accumulation',32),('max_steps',10),('max_grad_norm',1)]:
        cfg=deepcopy(suite);cfg['training'][key]=value
        with pytest.raises(ValueError):validate(cfg)
    cfg=deepcopy(suite);cfg['seeds']=[42,42]
    with pytest.raises(ValueError):validate(cfg)


def test_nested_sampling_is_independent_of_microbatch_and_no_replacement():
    a=selection(1000,8,42);b=selection(1000,64,42);c=selection(1000,512,42)
    assert a==b[:8] and b==c[:64] and len(set(c))==512
    assert c!=selection(1000,512,43)
    with pytest.raises(ValueError):selection(7,8,42)


@pytest.mark.parametrize('method',list(METHODS))
def test_all_methods_real_parameter_sharing_freezing_and_optimization(suite,method):
    torch.manual_seed(99)
    names=[str(i) for i in range(4)]
    gradients={n:torch.randn(8,8)*(1+int(n)//2)+int(n)//2 for n in names}
    suite['suite']['rank_budgets']['8']['total']=32
    if method=='lora':plan=[dict(family='q',layers=[n],rank=8) for n in names]
    else:plan,_=family_plan(gradients,names,'q',8,method,suite)
    model=nn.Sequential(*[nn.Linear(8,8,bias=False) for _ in names])
    if method!='lora':
        plan=active_rank_plan(plan,active_budgets(plan,suite))
        apply_partial_plan(model,plan,alpha=16,frozen_b_init_scale=.01)
    else:apply_plan(model,plan,alpha=16)
    counts=count_adapters(model,plan)
    assert counts['trainable_adapter_parameters']==sum(p.numel() for p in model.parameters() if p.requires_grad)
    for group in plan:
        first=model.get_submodule(group['layers'][0])
        assert all(model.get_submodule(n).down is first.down and model.get_submodule(n).up is first.up for n in group['layers'])
    state=adapter_state(model);fixed={n:x.clone() for n,x in state.items() if not x.requires_grad}
    trainable={n:x.clone() for n,x in state.items() if x.requires_grad}
    optimizer=torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],lr=.01,weight_decay=.1)
    for _ in range(2):
        optimizer.zero_grad();model(torch.randn(3,8)).square().mean().backward();optimizer.step()
    after=adapter_state(model)
    assert all(torch.equal(v,after[n]) for n,v in fixed.items())
    assert any(not torch.equal(v,after[n]) for n,v in trainable.items())
    if method=='rank_allocation_only':assert all(len(g['layers'])==1 for g in plan)
    if method=='sharing_only':assert all(g['rank']==8 for g in plan)


def formal_result(score=.5,seed=42,protocol='p',smoke=False):
    return dict(status='completed',smoke=smoke,eligible_for_formal_tables=not smoke,
        method='lora',base_rank=4,seed=seed,parameters={'trainable_adapter_parameters':3407872},
        protocol_sha256=protocol,checkpoint_sha256='hash'+str(seed),
        evaluation=dict(accuracy=score,evaluation_examples=1319,ordered_team_manifest_verified=True,
                        empty_outputs=0,unparseable_predictions=0))


def test_summary_excludes_smoke_deduplicates_and_does_not_mix_protocols(tmp_path):
    for name,result in [('a',formal_result()),('duplicate',formal_result()),('b',formal_result(.6,43)),
                        ('c',formal_result(.7,42,'other')),('smoke',formal_result(smoke=True))]:
        common.write_json(tmp_path/name/'result.json',result)
    rows,excluded=collect([tmp_path]);values=aggregate(rows)
    assert len(rows)==3 and len(excluded)==1
    assert sorted(r['n'] for r in values)==[1,2]
    assert next(r for r in values if r['n']==1)['std_percent'] is None
    report(rows,values,tmp_path,False)
    assert '55.00' in (tmp_path/'report.md').read_text()
    bad=formal_result();bad['checkpoint_sha256']='different'
    common.write_json(tmp_path/'conflict/result.json',bad)
    with pytest.raises(ValueError,match='Conflicting'):collect([tmp_path])


def test_source_archive_excludes_checkpoints_and_other_members(tmp_path):
    root=tmp_path/'repo'
    p=root/'experiments/chloe';p.mkdir(parents=True)
    (p/'train.py').write_text('print(1)\n');(p/'model.pt').write_bytes(b'weights')
    (root/'experiments/other').mkdir();(root/'experiments/other/private.py').write_text('private')
    shared=root/'src/e2lora/adapters';shared.mkdir(parents=True)
    from export_code import SHARED
    for name in SHARED:(shared/name).write_text('# test\n')
    archive=tmp_path/'source.zip';manifest=export(root,archive)
    assert not any(r['path'].endswith('.pt') or '/other/' in r['path'] for r in manifest)
    with zipfile.ZipFile(archive) as z:assert 'UPLOAD-MANIFEST.json' in z.namelist()


def test_retry_does_not_restart_cancelled_or_oom_jobs():
    assert retryable('TIMEOUT','0:0',True)
    assert retryable('FAILED','75:0',True)
    for state,code in [('CANCELLED','0:15'),('OUT_OF_MEMORY','0:9'),('FAILED','1:0'),('COMPLETED','0:0')]:
        assert not retryable(state,code,True)
    assert not retryable('TIMEOUT','0:0',False)


def test_smoke_gate_does_not_require_matching_paper_accuracy():
    result=dict(status='completed',smoke=True,fixed_components_unchanged=True,
        source_training_manifest_verified=True,evaluation=dict(evaluation_examples=8,accuracy=0,
        empty_outputs=0,unparseable_predictions=0))
    assert smoke_ok(result)
    result['evaluation']['empty_outputs']=1
    assert not smoke_ok(result)


@pytest.mark.parametrize('method',['lora','e2lora'])
def test_suite_training_checkpoint_resume_matches_uninterrupted_cpu_probe(suite,tmp_path,monkeypatch,method):
    """Exercise the new training loop, not a second implementation of it."""
    from contextlib import nullcontext
    from types import SimpleNamespace
    import sys
    import suite_train as driver
    import train
    class Attention(nn.Module):
        def __init__(self):
            super().__init__()
            for name in ['q_proj','k_proj','v_proj','o_proj']:setattr(self,name,nn.Linear(8,8,bias=False))
        def forward(self,x):
            return self.o_proj(torch.tanh(self.q_proj(x)+self.k_proj(x)+self.v_proj(x)))
    class Tiny(nn.Module):
        def __init__(self):
            super().__init__();self.config=SimpleNamespace(num_hidden_layers=4)
            self.model=nn.Module();self.model.layers=nn.ModuleList([nn.Module() for _ in range(4)])
            for layer in self.model.layers:layer.self_attn=Attention()
            self.embedding=nn.Embedding(16,8)
        def gradient_checkpointing_enable(self,**kw):pass
        def forward(self,input_ids,attention_mask,labels):
            x=self.embedding(input_ids)
            for layer in self.model.layers:x=layer.self_attn(x)
            return SimpleNamespace(loss=(x[:,:-1].square().sum(-1)*labels[:,1:].ne(-100)).sum()/labels[:,1:].ne(-100).sum())
    def loader(*args):return Tiny(),SimpleNamespace(pad_token_id=0)
    examples=[dict(input_ids=[1,2+i%8,3],labels=[-100,2+i%8,3]) for i in range(128)]
    monkeypatch.setattr(common,'load_local_model',loader)
    monkeypatch.setattr(driver,'data',lambda *args:(examples,examples[:2],{'train_indices':list(range(128))}))
    monkeypatch.setattr(driver,'validation_loss',lambda *args:.1)
    def evaluator(model,tokenizer,cfg,directory,smoke,epoch):
        value=dict(accuracy=.25,evaluation_examples=8 if smoke else 1319,ordered_team_manifest_verified=True,
                   empty_outputs=0,unparseable_predictions=0)
        common.write_json(directory/f'evaluation-epoch{epoch}.json',value);return value
    monkeypatch.setattr(driver,'evaluate',evaluator)
    monkeypatch.setattr(train,'require_gpu_allocation',lambda:None)
    monkeypatch.setattr(torch,'autocast',lambda *args,**kwargs:nullcontext())
    original_to=torch.Tensor.to
    def cpu_to(tensor,*args,**kwargs):
        if args and str(args[0]).startswith('cuda'):args=('cpu',)+args[1:]
        if str(kwargs.get('device','')).startswith('cuda'):kwargs['device']='cpu'
        return original_to(tensor,*args,**kwargs)
    monkeypatch.setattr(torch.Tensor,'to',cpu_to)
    monkeypatch.setattr(torch.cuda,'get_device_name',lambda *args:'CPU synthetic probe')
    monkeypatch.setattr(torch.cuda,'max_memory_allocated',lambda:0)
    monkeypatch.setattr(torch.cuda,'get_rng_state_all',lambda:[])
    monkeypatch.setattr(torch.cuda,'set_rng_state_all',lambda state:None)
    monkeypatch.setenv('SLURM_JOB_ID','cpu-test')
    monkeypatch.setattr(common,'OUTPUT_ROOT',tmp_path)
    suite['run']=dict(method=method,rank=8,seed=42,sampling_steps=0 if method=='lora' else 64,studies=['table5'])
    suite['suite']['rank_budgets']['8']['total']=32
    def invoke(name,resume=False,stop=False):
        cfg=deepcopy(suite);cfg['output_root']=str(tmp_path/name)
        config=tmp_path/(name+'.json');common.write_json(config,cfg)
        argv=['suite_train','--config',str(config),'--method',method,'--rank','8','--seed','42',
              '--sampling-steps',str(cfg['run']['sampling_steps']),'--smoke']
        if resume:argv.append('--resume')
        monkeypatch.setattr(sys,'argv',argv);driver.STOP=False
        original_write=common.write_json
        def writer(path,value):
            original_write(path,value)
            if stop and Path(path).name=='status.json' and value.get('step')==1:driver.STOP=True
        monkeypatch.setattr(driver,'write_json',writer)
        if stop:
            with pytest.raises(SystemExit) as e:driver.main()
            assert e.value.code==75
        else:driver.main()
        return next((tmp_path/name).rglob('checkpoint.pt'))
    full=invoke('full');invoke('resumed',stop=True);resumed=invoke('resumed',resume=True)
    left=torch.load(full,weights_only=False);right=torch.load(resumed,weights_only=False)
    assert left['step']==right['step']==2
    for name,value in left['adapter'].items():torch.testing.assert_close(value,right['adapter'][name],rtol=0,atol=0)
    assert left['history']==right['history']
