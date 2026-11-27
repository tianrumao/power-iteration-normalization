"""Evaluate frozen pilot checkpoints once on train/selection/confirmation/test splits."""
import argparse
import csv
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch


def main():
    p=argparse.ArgumentParser();p.add_argument('--run',type=Path,required=True);p.add_argument('--cache',type=Path,required=True);p.add_argument('--source',type=Path,required=True);a=p.parse_args()
    sys.path.insert(0,str(a.source))
    from src.datasets import dataset
    from src.training import model_for
    from src.metrics import score
    from src.io import atomic_json,write_csv
    torch.set_num_threads(1);device=torch.device('cuda')
    cfg=json.loads((a.run/'config.json').read_text())
    pilot=json.loads((a.run/'pilot-complete.json').read_text())
    chosen_path=a.run/'frozen-selection.json'
    if chosen_path.exists():
        chosen=json.loads(chosen_path.read_text());pairs=chosen['pairs'];mode='validation-band checkpoints frozen before independent confirmation'
    else:
        # A failed matching pilot still receives a transparent endpoint generalization audit.
        pairs=[];mode='unmatched endpoints; no common validation band was available'
        for seed in cfg['seeds']:
            pairs.append({'seed':seed,**{m:json.loads((a.run/f'seed-{seed}'/m/'history.json').read_text())[-1] for m in ['111','000']}})
    frozen={'selection_mode':mode,'pilot_status':pilot['status'],'pairs':pairs,
        'policy':'all checkpoints frozen before this audit; no optimizer steps; no test-driven checkpoint reselection'}
    audit_dir=a.run/'generalization-audit';audit_dir.mkdir(exist_ok=True)
    if (audit_dir/'complete.json').exists():
        print('GENERALIZATION_AUDIT_ALREADY_COMPLETE',flush=True);return
    atomic_json(audit_dir/'frozen-checkpoints.json',frozen)
    rows=[];gaps=[];pair_differences=[];hashes=[]
    for pair in pairs:
        seed=pair['seed'];pools=dataset(seed,cfg,a.run,a.cache)
        splits={'train':pools['train'],'selection_validation':tuple(v[:8192] for v in pools['validation']),
            'confirmation_validation':tuple(v[8192:] for v in pools['validation']),'test':pools['test']}
        pair_values={}
        for mask in ['111','000']:
            path=a.run/f'seed-{seed}'/mask/f'step-{pair[mask]["step"]:05d}.pt'
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            model=model_for(mask,cfg['init_std'],device,cfg['precision']);model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.eval().requires_grad_(False)
            arrays={}
            for name,pool in splits.items():
                metrics,errors=score(model,pool,device);values=1-errors;arrays[name]=values;pair_values[mask,name]=values
                se=float(values.std(ddof=1)/np.sqrt(len(values)))
                rows.append({'seed':seed,'mask':mask,'checkpoint_step':pair[mask]['step'],'split':name,'samples':len(values),
                    'mean_similarity':float(values.mean()),'standard_error':se,'approx_ci_low':max(0.,float(values.mean()-1.96*se)),
                    'approx_ci_high':min(1.,float(values.mean()+1.96*se)),'invalid_outputs':metrics['failures']})
                print(f'GENERALIZATION_SCORE seed={seed} mask={mask} step={pair[mask]["step"]} split={name} similarity={values.mean():.6f}',flush=True)
            np.savez_compressed(audit_dir/f'seed-{seed}-{mask}-sample-scores.npz',**arrays)
            train,test=arrays['train'],arrays['test'];gap=float(train.mean()-test.mean())
            se=float(np.sqrt(train.var(ddof=1)/len(train)+test.var(ddof=1)/len(test)))
            gaps.append({'seed':seed,'mask':mask,'train_minus_test':gap,'approx_ci_low':gap-1.96*se,'approx_ci_high':gap+1.96*se})
            assert hashlib.sha256(path.read_bytes()).hexdigest()==before
            hashes.append({'seed':seed,'mask':mask,'checkpoint_sha256':before,'unchanged':True})
            del model;torch.cuda.empty_cache()
        for split in splits:
            difference=pair_values['111',split]-pair_values['000',split];se=float(difference.std(ddof=1)/np.sqrt(len(difference)))
            pair_differences.append({'seed':seed,'split':split,'paired_111_minus_000':float(difference.mean()),
                'approx_ci_low':float(difference.mean()-1.96*se),'approx_ci_high':float(difference.mean()+1.96*se)})
        write_csv(audit_dir/'split-scores.csv',rows);write_csv(audit_dir/'train-test-gaps.csv',gaps);write_csv(audit_dir/'paired-split-differences.csv',pair_differences)
        del pools,splits,pair_values
    atomic_json(audit_dir/'complete.json',{'status':'complete','selection_mode':mode,'pilot_status':pilot['status'],
        'checkpoint_models':len(hashes),'test_scored_once_after_freezing':True,'training_performed':False,'checkpoint_audits':hashes,
        'interval_note':'Approximate per-example normal intervals conditional on each frozen model; these are not confidence intervals over training seeds.',
        'limitation':'A finite held-out audit reports evidence of generalization gaps; it cannot prove the absence of all overfitting or distribution-shift issues.'})
    print('GENERALIZATION_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
