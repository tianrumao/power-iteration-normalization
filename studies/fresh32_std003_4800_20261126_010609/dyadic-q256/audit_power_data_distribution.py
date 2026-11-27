"""Audit generation and convergence heterogeneity without modifying data or fitting models."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--source', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    sys.path.insert(0, str(a.source))
    from src.data import sample
    from src.power_targets import normalize, power_directions
    a.output.mkdir(parents=True, exist_ok=True)
    data = sample(2048, 32, 64, np.random.default_rng(20261125), power_steps=0)
    x, start = data.tokens[:, :32, :-1], data.tokens[:, 32:, -1]
    cov = x @ x.swapaxes(-1, -2)
    val, vec = np.linalg.eigh(cov)
    truth = vec[:, :, -1]
    alpha = np.einsum('nd,nd->n', start, truth)**2
    ratio = val[:, -2]/val[:, -1]
    directions = power_directions(data.tokens, 256)
    # Independent reference: exponentiation in the eigenbasis avoids iterative accumulation.
    coeff = np.einsum('ndi,nd->ni', vec, start)
    powers = [2**i for i in range(9)]
    samples, distributions, worst = [], [], 0.
    f32 = power_directions(data.tokens.astype(np.float32), 256)
    initial = alpha.copy()
    for k in powers:
        score = np.einsum('nd,nd->n', directions[:, k], truth).clip(-1,1)**2
        spectral = normalize(np.einsum('ndi,ni->nd', vec, coeff*(val/val[:, -1, None])**k))
        worst = max(worst, float(np.max(1-np.einsum('nd,nd->n', spectral,directions[:, k])**2)))
        f32score = np.einsum('nd,nd->n', f32[:, k], truth).clip(-1,1)**2
        distributions.append({'power': k, 'mean': float(score.mean()), 'median': float(np.median(score)),
            'p10': float(np.quantile(score,.1)), 'p25': float(np.quantile(score,.25)),
            'p75': float(np.quantile(score,.75)), 'p90': float(np.quantile(score,.9)),
            'fraction_below_0_5': float((score<.5).mean()), 'fraction_ge_0_99': float((score>=.99).mean()),
            'float32_input_mean_score_difference': float(np.mean(f32score-score))})
        for i,v in enumerate(score):
            samples.append({'sample': i, 'power': k, 'similarity': float(v), 'lambda2_over_lambda1': float(ratio[i]),
                'initial_squared_alignment': float(alpha[i])})
    all_scores=np.einsum('nkd,nd->nk', directions, truth)**2
    report = {'samples': 2048, 'seed':20261125, 'd':32, 'n':64,
        'distribution':'independent unit-column Gaussian X and independent isotropic Gaussian start a',
        'column_unit_norm_max_error':float(np.max(np.abs(np.linalg.norm(x,axis=1)-1))),
        'start_unit_norm_max_error':float(np.max(np.abs(np.linalg.norm(start,axis=1)-1))),
        'minimum_eigenvalue':float(val.min()),
        'truth_max_relative_eigen_residual':float(np.max(np.linalg.norm(np.einsum('nij,nj->ni',cov,truth)-val[:,-1,None]*truth,axis=1)/val[:,-1])),
        'independent_spectral_vs_iterative_max_direction_error':worst,
        'monotonicity_violations_gt_1e_10':int((np.diff(all_scores,axis=1)<-1e-10).sum()),
        'ratio_quantiles':np.quantile(ratio,[.01,.1,.5,.9,.99]).tolist(),
        'initial_alignment_quantiles':np.quantile(initial,[.01,.1,.5,.9,.99]).tolist(),
        'initial_alignment_mean':float(initial.mean()), 'random_direction_expected_mean':1/32,
        'powers':distributions, 'strata':[],
        'target_pairwise_mean_squared_cosine': np.einsum('nid,njd->nij',directions[:,powers],directions[:,powers]).clip(-1,1).__pow__(2).mean(0).tolist()}
    q16=all_scores[:,16]
    for label, variable in [('spectral_ratio',ratio),('initial_alignment',alpha)]:
        edges=np.quantile(variable,np.linspace(0,1,5))
        for j in range(4):
            keep=(variable>=edges[j]) & ((variable<=edges[j+1]) if j==3 else (variable<edges[j+1]))
            report['strata'].append({'variable':label,'quartile':j+1,'low':float(edges[j]),'high':float(edges[j+1]),
                'n':int(keep.sum()),'q16_mean':float(q16[keep].mean()),'q16_median':float(np.median(q16[keep]))})
    (a.output/'data-distribution-audit.json').write_text(json.dumps(report,indent=2))
    for name,rows in [('data-distribution-summary.csv',distributions),('data-distribution-samples.csv',samples)]:
        with (a.output/name).open('w') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    print(json.dumps(report,indent=2))


if __name__=='__main__': main()
