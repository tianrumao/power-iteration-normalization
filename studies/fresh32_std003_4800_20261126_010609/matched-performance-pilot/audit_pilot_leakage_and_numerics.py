"""Check matrix overlap across data splits and frozen-model numerical health."""
import argparse
import hashlib
import itertools
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
    from src.io import atomic_json
    cfg=json.loads((a.run/'config.json').read_text());torch.set_num_threads(1);device=torch.device('cuda')
    frozen=json.loads((a.run/'generalization-audit/frozen-checkpoints.json').read_text())
    all_records=[]
    for pair in frozen['pairs']:
        seed=pair['seed'];pools=dataset(seed,cfg,a.run,a.cache)
        splits={'train':pools['train'][0],'selection_validation':pools['validation'][0][:8192],
            'confirmation_validation':pools['validation'][0][8192:],'test':pools['test'][0]}
        hashes={}
        for name,inputs in splits.items():
            # Hash X, not (X,a): the supervised eigenvector depends only on X.
            hashes[name]={hashlib.sha256(row[:cfg['d'],:-1].numpy().tobytes()).digest() for row in inputs}
        overlaps=[{'split_a':x,'split_b':y,'shared_matrices':len(hashes[x]&hashes[y])} for x,y in itertools.combinations(splits,2)]
        assert all(r['shared_matrices']==0 for r in overlaps),'Matrix overlap across training/held-out splits'
        unique={name:{'samples':len(inputs),'unique_matrices':len(hashes[name])} for name,inputs in splits.items()}
        assert all(v['samples']==v['unique_matrices'] for v in unique.values()),'Duplicated matrices within a split'
        checks=[]
        for mask in ['111','000']:
            step=pair[mask]['step'];path=a.run/f'seed-{seed}'/mask/f'step-{step:05d}.pt'
            before=hashlib.sha256(path.read_bytes()).hexdigest()
            model=model_for(mask,cfg['init_std'],device,cfg['precision']);model.load_state_dict(torch.load(path,map_location=device,weights_only=True));model.eval().requires_grad_(False)
            x=splits['test'][:128].to(device=device,dtype=torch.float64)
            with torch.no_grad():states,_=model(x)
            layer_records=[]
            for layer,state in enumerate([x,*states]):
                values=state.cpu().numpy();finite=np.isfinite(values)
                assert finite.all(),'Nonfinite intermediate coordinates in frozen-model audit'
                peak=np.max(np.abs(values),axis=-2,keepdims=True)
                okay=peak>0;scaled=values/np.where(okay,peak,1)
                length=np.linalg.norm(scaled,axis=-2,keepdims=True)
                log_norm=np.log10(np.where(okay,peak,1))+np.log10(np.where(length>0,length,1))
                valid=okay & (length>0)
                layer_records.append({'layer':layer,'samples':len(values),'finite_coordinates':bool(finite.all()),
                    'max_log10_absolute_coordinate':float(np.log10(np.max(np.abs(values)))) if np.any(values) else None,
                    'median_log10_column_norm':float(np.median(log_norm[valid])) if valid.any() else None,
                    'max_log10_column_norm':float(log_norm[valid].max()) if valid.any() else None,'zero_columns':int((~valid).sum())})
            assert hashlib.sha256(path.read_bytes()).hexdigest()==before
            checks.append({'mask':mask,'checkpoint_step':step,'checkpoint_sha256':before,'unchanged':True,'layers':layer_records})
            del model,states,x;torch.cuda.empty_cache()
        all_records.append({'seed':seed,'splits':unique,'overlaps':overlaps,'models':checks})
        del pools,splits,hashes
        print('LEAKAGE_NUMERICS_AUDITED',seed,flush=True)
    atomic_json(a.run/'generalization-audit/leakage-and-numerics.json',{'status':'complete',
        'matrix_hash_policy':'exact SHA-256 of encoded X, excluding the independently random start a',
        'training_performed':False,'numerical_scope':'all coordinates in input and four layer states on the first 128 held-out test samples per frozen model',
        'records':all_records,'limitation':'Exact-duplicate and finite-coordinate checks do not rule out all forms of statistical overfitting or distribution shift.'})
    print('LEAKAGE_NUMERICS_AUDIT_COMPLETE',flush=True)


if __name__=='__main__':main()
