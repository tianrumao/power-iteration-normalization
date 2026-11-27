"""Audit probe convergence and compare both target definitions on identical examples."""
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch


def read_csv(path):
    with path.open() as handle:
        return list(csv.DictReader(handle))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    assert json.loads((args.output / 'complete.json').read_text())['status'] == 'complete'
    sys.path.insert(0, str(args.source))
    from src.data import sample
    from src.power_targets import power_directions
    from src.io import atomic_json, digest, write_csv
    from src.training import model_for
    from src.probe_math import collect_query_states
    from src.direction_probes import partition

    cfg = json.loads((args.output / 'config.json').read_text())
    torch.set_num_threads(1)
    scores = read_csv(args.output / 'power-probes.csv')
    old_scores = read_csv(args.parent / 'power-probes.csv')
    keys = ('seed', 'mask', 'layer', 'power', 'split_seed')
    old = {tuple(r[k] for k in keys): r for r in old_scores}
    assert len(scores) == len(old) == 7680
    comparison = []
    for r in scores:
        o = old[tuple(r[k] for k in keys)]
        comparison.append({**{k: r[k] for k in keys},
            'old_similarity': float(o['similarity']), 'new_similarity': float(r['similarity']),
            'change': float(r['similarity']) - float(o['similarity']),
            'old_best_step': int(o['best_step']), 'new_best_step': int(r['best_step']),
            'validation_gain': float(r['validation_similarity']) - float(o['validation_similarity'])})
    write_csv(args.output / 'budget-comparison.csv', comparison)
    convergence = []
    for seed in cfg['seeds']:
        for mask in cfg['masks']:
            for split in cfg['probe_split_seeds']:
                fitted = np.load(args.output / f'seed-{seed}' / mask / f'split-{split}.npz')
                history = fitted['history_similarity']
                steps = fitted['history_steps']
                before = history[steps <= 900].max(0)
                after = history.max(0)
                for layer in range(5):
                    for index, power in enumerate(fitted['powers']):
                        convergence.append({'seed': seed, 'mask': mask, 'split_seed': split,
                            'layer': layer, 'power': int(power),
                            'late_validation_gain_900_to_1200': float(after[layer, index] - before[layer, index]),
                            'best_step': int(fitted['best_step'][layer, index])})
    write_csv(args.output / 'convergence.csv', convergence)
    target_rows = []
    data_dir = args.output / 'verified-probe-data'
    data_dir.mkdir(exist_ok=True)
    for seed in cfg['seeds']:
        manifest = json.loads((args.parent / f'seed-{seed}-data.json').read_text())
        stream = np.random.SeedSequence([cfg['data_epoch'], seed, 10723]).spawn(4)[3]
        generated = sample(cfg['probe_samples'], cfg['d'], cfg['n'], np.random.default_rng(stream), power_steps=0)
        inputs = generated.tokens.astype(np.float32)
        directions = power_directions(inputs, cfg['max_power'])
        assert digest(inputs) == manifest['splits']['probe']['input_sha256']
        assert digest(directions) == manifest['power_sha256']
        np.savez_compressed(data_dir / f'seed-{seed}.npz',
                            inputs=inputs, power_directions=directions,
                            eigenvector_reference=generated.target)
        for mask in cfg['masks']:
            model = model_for(mask, cfg['init_std'], torch.device('cuda'), cfg['precision'])
            model.load_state_dict(torch.load(args.parent / f'seed-{seed}' / mask / 'selected.pt',
                                            map_location='cuda', weights_only=True))
            model.eval().requires_grad_(False)
            _, fixed = collect_query_states(model, inputs, cfg['batch_size'], torch.device('cuda'))
            for split in cfg['probe_split_seeds']:
                _, _, test = partition(len(inputs), split)
                prediction = fixed[4, test]
                eig = generated.target[test]
                q16 = directions[test, 16]
                sq = lambda a, b: np.einsum('nd,nd->n', a, b).clip(-1, 1) ** 2
                same_q16 = float(sq(prediction, q16).mean())
                original = next(r for r in old_scores if int(r['seed']) == seed and r['mask'] == mask
                                and int(r['layer']) == 4 and int(r['power']) == 16 and int(r['split_seed']) == split)
                assert abs(same_q16 - float(original['fixed_similarity'])) < 1e-10
                target_rows.append({'seed': seed, 'mask': mask, 'split_seed': split,
                    'same_examples_fixed_vs_q16': same_q16,
                    'same_examples_fixed_vs_true_eigenvector': float(sq(prediction, eig).mean()),
                    'same_examples_q16_vs_true_eigenvector': float(sq(q16, eig).mean()),
                    'mean_second_to_first_eigenvalue_ratio': float(generated.eigenvalue_ratio[test].mean())})
            del model, fixed
            torch.cuda.empty_cache()
        write_csv(args.output / 'same-example-target-audit.csv', target_rows)
        print(f'TARGET_AUDIT_COMPLETE seed={seed}', flush=True)
        del generated, inputs, directions
    late = [r['late_validation_gain_900_to_1200'] for r in convergence if r['layer'] in (1, 2, 3)]
    atomic_json(args.output / 'audit-complete.json', {
        'status': 'complete', 'comparison_rows': len(comparison), 'target_audit_rows': len(target_rows),
        'verified_probe_pools': len(cfg['seeds']),
        'probe_data_note': 'Regenerated inputs and power labels exactly match saved parent SHA256; eigenvector references are newly recomputed.',
        'intermediate_layer_at_cap': sum(r['best_step'] == 1200 for r in convergence if r['layer'] in (1, 2, 3)),
        'intermediate_layer_maps': len(late),
        'late_validation_gain_mean': float(np.mean(late)),
        'late_validation_gain_p95': float(np.quantile(late, .95))})
    print('PROBE_TARGET_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
