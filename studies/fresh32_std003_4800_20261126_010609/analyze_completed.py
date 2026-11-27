import csv, json
from pathlib import Path
import numpy as np

ROOT=Path(__file__).resolve().parent
DATA=ROOT/'completed-results'
def read(name):
    with (DATA/name).open() as f: return list(csv.DictReader(f))
probes=read('power-probes.csv'); bypass=read('bypass.csv'); gaps=read('seed-gaps.csv')
seeds=sorted({int(r['seed']) for r in probes})
assert len(probes)==7680 and len(bypass)==512 and len(seeds)==32
assert len({(r['seed'],r['mask'],r['layer'],r['power'],r['split_seed']) for r in probes})==7680
flagged={int(r['seed']) for r in gaps if float(r['final_gap'])>.1}
P={}
for r in probes:
    key=(int(r['seed']),r['mask'],int(r['layer']),int(r['power']))
    P.setdefault(key,[]).append(r)
M={k:{metric:float(np.mean([float(r[metric]) for r in rows])) for metric in ['similarity','corrected_similarity','mean_error','shuffled_similarity','fixed_similarity','target_specificity_margin']} for k,rows in P.items()}
rng=np.random.default_rng(20261125)
def stats(values):
    a=np.asarray(values); n=len(a)
    boot=a[rng.integers(n,size=(20000,n))].mean(1)
    return dict(n=n,mean=float(a.mean()),sd=float(a.std(ddof=1)),ci=[float(x) for x in np.quantile(boot,[.025,.975])],positive=int((a>0).sum()))
cohorts={}
for name,ss in [('all',seeds),('gap_le_01',[s for s in seeds if s not in flagged])]:
    summary=[]
    for mask in ['111','000']:
        for layer in range(5):
            for power in [1,2,3,4,6,8,11,16]:
                row=dict(mask=mask,layer=layer,power=power)
                for metric in ['similarity','corrected_similarity','shuffled_similarity','fixed_similarity','target_specificity_margin']:
                    row[metric]=stats([M[s,mask,layer,power][metric] for s in ss])
                summary.append(row)
    best=[]
    for layer in range(5):
        for mask in ['111','000']:
            rows=[r for r in summary if r['layer']==layer and r['mask']==mask]
            winner=max(rows,key=lambda r:r['similarity']['mean'])
            wins={}
            for s in ss:
                p=max([1,2,3,4,6,8,11,16],key=lambda p:M[s,mask,layer,p]['similarity'])
                wins[str(p)]=wins.get(str(p),0)+1
            best.append(dict(layer=layer,mask=mask,power=winner['power'],similarity=winner['similarity'],seed_winners=wins))
    contrasts=[]
    for layer in range(5):
        for power in [1,2,3,4,6,8,11,16]:
            contrasts.append(dict(layer=layer,power=power,**stats([M[s,'111',layer,power]['similarity']-M[s,'000',layer,power]['similarity'] for s in ss])))
    by=[]; tests=[]; prim=[]
    B={(int(r['seed']),r['mask'],r['mode'],int(r['layer'])):r for r in bypass}
    for mask in ['111','000']:
        tests.append(dict(mask=mask,**stats([float(B[s,mask,'bypass',1]['intact_error']) for s in ss])))
        for mode in ['bypass','matched']:
            for layer in range(1,5):
                by.append(dict(mask=mask,mode=mode,layer=layer,**stats([float(B[s,mask,mode,layer]['delta_final_error']) for s in ss])))
    for mode in ['bypass','matched']:
        vals=[]; valsrel=[]
        for s in ss:
            terms={mask:float(B[s,mask,mode,4]['delta_final_error'])-float(B[s,mask,mode,1]['delta_final_error']) for mask in ['111','000']}
            vals.append(terms['111']-terms['000'])
            valsrel.append(terms['111']/(1-float(B[s,'111',mode,1]['intact_error']))-terms['000']/(1-float(B[s,'000',mode,1]['intact_error'])))
        prim.append(dict(mode=mode,contrast=stats(vals),relative_sensitivity=stats(valsrel)))
    cohorts[name]=dict(n=len(ss),summary=summary,best=best,probe_contrasts=contrasts,bypass=by,test_error=tests,primary=prim)
result=dict(cohorts=cohorts,flagged=sorted(flagged),gaps=[dict(seed=int(r['seed']),gap=float(r['final_gap'])) for r in gaps],counts=dict(probe_rows=len(probes),bypass_rows=len(bypass)),method='Split seeds averaged within model; paired model seeds are sampling unit; percentile bootstrap 20,000 resamples; exploratory unadjusted 95% CI.')
(DATA/'analysis.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps({name:{k:v for k,v in c.items() if k in ['best','test_error','primary']} for name,c in cohorts.items()},indent=2))
