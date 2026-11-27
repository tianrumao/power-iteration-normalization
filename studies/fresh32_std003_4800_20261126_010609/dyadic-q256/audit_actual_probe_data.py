"""Check all saved probe inputs and labels, preserving every original array."""
import argparse
import csv
import json
from pathlib import Path
import numpy as np


def main():
    p=argparse.ArgumentParser();p.add_argument('--data',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    sample_rows=[];audits=[]
    for path in sorted(a.data.glob('seed-*.npz')):
        seed=int(path.stem.split('-')[-1])
        with np.load(path) as z:
            tokens=z['inputs'].astype(np.float64);q=z['power_directions'];reference=z['eigenvector_reference']
        x,start=tokens[:,:32,:-1],tokens[:,32:,-1]
        assert np.isfinite(tokens).all() and np.isfinite(q).all() and np.isfinite(reference).all()
        covariance=x@x.swapaxes(-1,-2)
        ev,u=np.linalg.eigh(covariance);truth=u[:,:,-1]
        ratio=ev[:,-2]/ev[:,-1];alpha=np.einsum('nd,nd->n',start,truth)**2
        reference_score=np.einsum('nd,nd->n',reference,truth).clip(-1,1)**2
        q16_score=np.einsum('nd,nd->n',q[:,16],truth).clip(-1,1)**2
        coeff=np.einsum('ndi,nd->ni',u,start)*(ev/ev[:,-1,None])**16
        exact=np.einsum('ndi,ni->nd',u,coeff);exact/=np.linalg.norm(exact,axis=-1,keepdims=True)
        label_error=1-np.einsum('nd,nd->n',exact,q[:,16]).clip(-1,1)**2
        audit={'seed':seed,'samples':len(x),'finite':True,'unit_column_max_error':float(np.max(np.abs(np.linalg.norm(x,axis=1)-1))),
            'start_norm_max_error':float(np.max(np.abs(np.linalg.norm(start,axis=1)-1))),
            'zero_padding_max_absolute':float(max(np.max(np.abs(tokens[:,32:,:-1])),np.max(np.abs(tokens[:,:32,-1])))),
            'minimum_eigenvalue':float(ev.min()),'reference_max_direction_error':float(np.max(1-reference_score)),
            'q16_max_spectral_direction_error':float(label_error.max()),'q16_mean':float(q16_score.mean()),'q16_median':float(np.median(q16_score))}
        assert audit['zero_padding_max_absolute']==0 and audit['minimum_eigenvalue']>0
        assert audit['reference_max_direction_error']<1e-8 and audit['q16_max_spectral_direction_error']<1e-10
        audits.append(audit)
        for i in range(len(x)):
            sample_rows.append({'seed':seed,'sample':i,'lambda2_over_lambda1':float(ratio[i]),'initial_squared_alignment':float(alpha[i]),'q16_similarity_to_truth':float(q16_score[i])})
        print('ACTUAL_POOL_AUDITED',seed,flush=True)
    assert len(audits)==32 and len(sample_rows)==96000
    score=np.array([r['q16_similarity_to_truth'] for r in sample_rows]);ratio=np.array([r['lambda2_over_lambda1'] for r in sample_rows]);alpha=np.array([r['initial_squared_alignment'] for r in sample_rows])
    out={'status':'complete','seeds':len(audits),'samples':len(sample_rows),'original_arrays_modified':False,
        'note':'All actual saved float32 inputs; independent float64 eigendecomposition and spectral q16 check. Saved references came from pre-cast float64 generator inputs.',
        'q16_mean':float(score.mean()),'q16_median':float(np.median(score)),'q16_p10':float(np.quantile(score,.1)),
        'ratio_quantiles':np.quantile(ratio,[.1,.5,.9]).tolist(),'initial_alignment_quantiles':np.quantile(alpha,[.1,.5,.9]).tolist(),'audits':audits}
    a.output.mkdir(parents=True,exist_ok=True)
    (a.output/'actual-probe-data-audit.json').write_text(json.dumps(out,indent=2))
    with (a.output/'actual-probe-data-difficulty.csv').open('w') as f:
        w=csv.DictWriter(f,fieldnames=list(sample_rows[0]));w.writeheader();w.writerows(sample_rows)
    print(json.dumps({k:v for k,v in out.items() if k!='audits'},indent=2))


if __name__=='__main__':main()
