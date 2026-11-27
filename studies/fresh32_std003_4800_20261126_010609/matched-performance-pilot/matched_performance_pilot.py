"""Independent pilot for validation-matched mother models; no probes or test selection."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser()
    p.add_argument('--parent',type=Path,required=True)
    p.add_argument('--source',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,required=True)
    p.add_argument('--learning-rate',type=float,default=.001)
    p.add_argument('--max-steps',type=int,default=12800)
    p.add_argument('--seeds',type=int,nargs='+',default=[9003,9004])
    a=p.parse_args();sys.path.insert(0,str(a.source))
    from src.datasets import dataset
    from src.training import model_for,cpu_state,learning_rate
    from src.model import readout,direction_error
    from src.metrics import score
    from src.io import atomic_json,digest
    assert torch.cuda.is_available()
    torch.set_num_threads(1);device=torch.device('cuda')
    a.output.mkdir(parents=True,exist_ok=True);a.cache.mkdir(parents=True,exist_ok=True)
    cfg=json.loads((a.parent/'config.json').read_text())
    cfg.update(seeds=a.seeds,learning_rate=a.learning_rate,budgets=[a.max_steps],schedule_horizon=a.max_steps,
        validation_size=16384,test_size=8192,probe_steps=1200,checkpoint_selection='independent validation performance band',
        primary_gap_tolerance=.01,sensitivity_gap_tolerance=.025)
    protocol={'status':'pilot','parent_run':a.parent.name,'independent_pilot_seeds':a.seeds,
        'data_generator':'original unit-column Gaussian X, independent random a; unchanged model and encoding',
        'normalizer':'L2 normalization of token columns after blocks 1-3; not standard LayerNorm',
        'pairing':'same data, initialization, batches, precision, learning rate and schedule within each pair',
        'changed_training':'up to 12800 updates, pilot learning rate .001, dense saved intermediate weights',
        'candidate_common_similarities':[.78,.75],'candidate_band_half_width':.005,
        'primary_gap_tolerance':.01,'sensitivity_gap_tolerance':.025,
        'selection_validation_samples':8192,'independent_confirmation_validation_samples':8192,
        'test_samples':8192,'test_policy':'not loaded or scored by this pilot; never used for checkpoint selection',
        'probe_policy':'no probe fitting in the pilot; full cohort only after a passing gate and a frozen common target',
        'positive_early_layer_effect_required':False,
        'failure_policy':'retain all histories and checkpoints; do not call an unmatched pair matched',
        'raw_data_storage':str(a.cache),'config':cfg}
    if (a.output/'protocol.json').exists():
        assert json.loads((a.output/'protocol.json').read_text())==protocol,'Use a separate output for protocol changes'
    atomic_json(a.output/'protocol.json',protocol);atomic_json(a.output/'config.json',cfg)
    histories={};pairing=[]
    for seed in a.seeds:
        print('PILOT_DATA_START',seed,flush=True)
        pools=dataset(seed,cfg,a.output,a.cache)
        # The second validation half is not read until all pilot checkpoint choices are fixed.
        selection=(pools['validation'][0][:8192],pools['validation'][1][:8192])
        for mask in ['111','000']:
            folder=a.output/f'seed-{seed}'/mask;folder.mkdir(parents=True,exist_ok=True)
            torch.manual_seed(seed);model=model_for(mask,cfg['init_std'],device,cfg['precision'])
            init=digest(np.concatenate([v.numpy().ravel() for v in cpu_state(model).values()]))
            optimizer=torch.optim.Adam(model.parameters(),lr=a.learning_rate)
            rng=np.random.default_rng(np.random.SeedSequence([cfg['data_epoch'],seed,4402]))
            batches=rng.integers(0,cfg['train_size'],(a.max_steps,cfg['batch_size']))
            record={'seed':seed,'mask':mask,'initial_sha256':init,'batch_sha256':digest(batches)}
            pairing.append(record);atomic_json(folder/'pairing.json',record)
            history=[];start=0;stable=0;resume=folder/'resume.pt'
            if resume.exists():
                saved=torch.load(resume,map_location=device,weights_only=True)
                model.load_state_dict(saved['model']);optimizer.load_state_dict(saved['optimizer'])
                history=saved['history'];start=saved['step'];stable=saved['stable']
            x,y=(v.to(device=device,dtype=torch.float64) for v in pools['train'])
            if not history:
                result,_=score(model,selection,device)
                history.append({'step':0,**result})
            for step in range(start+1,a.max_steps+1):
                if stable>=2:break
                optimizer.param_groups[0]['lr']=learning_rate(step-1,cfg)
                ids=batches[step-1];optimizer.zero_grad(set_to_none=True)
                states,_=model(x[ids]);prediction,valid=readout(states[-1])
                loss=direction_error(prediction,y[ids],valid).mean()
                if not valid.all() or not torch.isfinite(loss):raise FloatingPointError('Invalid pilot output')
                loss.backward()
                if not all(torch.isfinite(w.grad).all() for w in model.parameters()):raise FloatingPointError('Invalid pilot gradient')
                optimizer.step()
                # Finer checkpoint spacing near the performance bands avoids artificial mismatch.
                interval=50 if history[-1]['mean_similarity']>=.72 else 200
                if step%interval==0 or step==a.max_steps:
                    result,_=score(model,selection,device);history.append({'step':step,**result})
                    checkpoint=folder/f'step-{step:05d}.pt'
                    torch.save(cpu_state(model),checkpoint)
                    stable=stable+1 if result['mean_similarity']>=.785 else 0
                    atomic_json(folder/'history.json',history)
                    torch.save({'model':cpu_state(model),'optimizer':optimizer.state_dict(),'step':step,
                        'history':history,'stable':stable},folder/'resume.pending')
                    (folder/'resume.pending').replace(resume)
                    print(f'PILOT_EVAL seed={seed} mask={mask} step={step} selection_similarity={result["mean_similarity"]:.6f}',flush=True)
            histories[seed,mask]=history
            del model,optimizer,x,y;torch.cuda.empty_cache()
            print('PILOT_ARM_COMPLETE',seed,mask,flush=True)
        assert pairing[-2]['initial_sha256']==pairing[-1]['initial_sha256']
        assert pairing[-2]['batch_sha256']==pairing[-1]['batch_sha256']
        del pools
    # Freeze the common target and all checkpoints before reading confirmation validation.
    chosen=None
    for level in [.78,.75]:
        pairs=[]
        for seed in a.seeds:
            candidates={mask:[r for r in histories[seed,mask] if r['step']>0 and abs(r['mean_similarity']-level)<=.005] for mask in ['111','000']}
            possibilities=[(on,off) for on in candidates['111'] for off in candidates['000'] if abs(on['mean_similarity']-off['mean_similarity'])<=.01]
            if not possibilities:break
            on,off=min(possibilities,key=lambda pair:(max(abs(r['mean_similarity']-level) for r in pair),abs(pair[0]['mean_similarity']-pair[1]['mean_similarity']),sum(r['step'] for r in pair)))
            pairs.append({'seed':seed,'111':on,'000':off})
        if len(pairs)==len(a.seeds):chosen={'level':level,'pairs':pairs};break
    if chosen is None:
        result={'status':'pilot_failed','reason':'no common performance band reached by both conditions in every pilot seed','test_scored':False,'probes_fitted':False}
        atomic_json(a.output/'pilot-complete.json',result);print('PILOT_GATE_FAILED '+json.dumps(result),flush=True);return
    atomic_json(a.output/'frozen-selection.json',chosen)
    results=[]
    for pair in chosen['pairs']:
        seed=pair['seed'];pools=dataset(seed,cfg,a.output,a.cache)
        confirmation=(pools['validation'][0][8192:],pools['validation'][1][8192:])
        arms={}
        for mask in ['111','000']:
            folder=a.output/f'seed-{seed}'/mask;step=pair[mask]['step'];path=folder/f'step-{step:05d}.pt'
            model=model_for(mask,cfg['init_std'],device,cfg['precision'])
            model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.eval().requires_grad_(False)
            metrics,_=score(model,confirmation,device)
            arms[mask]={'selected_step':step,'selection_similarity':pair[mask]['mean_similarity'],
                'confirmation_similarity':metrics['mean_similarity'],'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            del model;torch.cuda.empty_cache()
        gap=abs(arms['111']['confirmation_similarity']-arms['000']['confirmation_similarity'])
        results.append({'seed':seed,'arms':arms,'confirmation_gap':gap,'passes_primary':gap<=.01,'passes_sensitivity':gap<=.025})
        del pools
    passed=all(r['passes_primary'] for r in results)
    result={'status':'pilot_passed' if passed else 'pilot_failed','common_similarity_level':chosen['level'],
        'primary_tolerance':.01,'sensitivity_tolerance':.025,'pairs':results,'test_scored':False,'probes_fitted':False,
        'next_step':'freeze the recipe and common target before starting independent full-cohort seeds' if passed else 'inspect training and confirmation mismatch before launching a full cohort'}
    atomic_json(a.output/'pilot-complete.json',result)
    print(('PILOT_GATE_PASSED ' if passed else 'PILOT_GATE_FAILED ')+json.dumps(result),flush=True)


if __name__=='__main__':main()
