"""Frozen-model q256 probes completing the preselected dyadic grid q1 through q256."""
import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch

LABELS = ['q256']
POWERS = [256]
CALIBRATION_POWERS = [2**j for j in range(9)]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--probe-data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    args = parser.parse_args()
    args.source.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(args.parent / 'source.zip') as archive:
        archive.extractall(args.source)
    sys.path.insert(0, str(args.source))
    from src.data import sample
    from src.power_targets import power_directions, normalize
    from src.io import atomic_json, digest, write_csv
    from src.training import model_for
    from src.probe_math import collect_query_states
    from src import direction_probes as dp
    from probe_fit_trace import fit_maps_trace

    assert torch.cuda.is_available(), 'A GPU runtime is required.'
    torch.set_num_threads(1)
    device = torch.device('cuda')
    cfg = json.loads((args.parent / 'config.json').read_text())
    settings = dict(steps=1200, batch_size=cfg['probe_batch_size'], learning_rate=cfg['probe_learning_rate'])
    args.output.mkdir(parents=True, exist_ok=True)
    protocol = {'parent_run': args.parent.name, 'mother_models_retrained': False,
        'targets': LABELS, 'powers': POWERS, 'full_display_grid': CALIBRATION_POWERS, 'probe': settings,
        'features': 'full 2*d query token at input 0 and layers 1-4',
        'splits': cfg['probe_split_seeds'], 'selection': 'validation only; fixed and random initialization',
        'data_source': str(args.probe_data),
        'true_target_source': 'saved eigenvector_reference from the verified parent probe-data audit',
        'high_power_source': 'power iteration on the same encoded X and initial vector, normalized every step',
        'negative_control': 'permute fit and validation labels independently; score against true test labels',
        'versions': {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__)}}
    if (args.output / 'protocol.json').exists():
        prior = json.loads((args.output / 'protocol.json').read_text())
        for key in ('parent_run', 'targets', 'probe', 'features', 'splits'):
            assert prior[key] == protocol[key], f'Cannot mix studies: {key}'
    atomic_json(args.output / 'protocol.json', protocol)
    atomic_json(args.output / 'parent-config.json', cfg)
    with (args.probe_data.parent / 'same-example-target-audit.csv').open() as handle:
        prior_eigen_scores = {(int(r['seed']), r['mask'], int(r['split_seed'])):
            float(r['same_examples_fixed_vs_true_eigenvector']) for r in csv.DictReader(handle)}
    if not (args.output / 'calibration.json').exists():
        calibration = sample(cfg['calibration_samples'], cfg['d'], cfg['n'],
            np.random.default_rng(cfg['calibration_seed']), power_steps=0)
        directions = power_directions(calibration.tokens.astype(np.float32), max(POWERS))
        rows = []
        for power in CALIBRATION_POWERS:
            values = np.einsum('nd,nd->n', directions[:, power], calibration.target).clip(-1, 1) ** 2
            rows.append({'power': power, 'samples': len(values), 'mean_similarity_to_eigenvector': float(values.mean()),
                'p10_similarity': float(np.quantile(values, .1)), 'median_similarity': float(np.median(values)),
                'p90_similarity': float(np.quantile(values, .9)), 'fraction_ge_0_99': float((values >= .99).mean()),
                'fraction_ge_0_999': float((values >= .999).mean())})
        atomic_json(args.output / 'calibration.json', {'seed': cfg['calibration_seed'], 'rows': rows,
            'purpose': 'independent descriptive convergence check; powers 2^0 through 2^8 fixed before fitting q256'})
        write_csv(args.output / 'calibration.csv', rows)
        print('CALIBRATION_COMPLETE ' + json.dumps(rows), flush=True)
        del calibration, directions

    def summarize(rows, matched=False):
        parent_pairs = json.loads((args.parent / 'pairs.json').read_text())
        included = {p['seed'] for p in parent_pairs if not matched or not p['gap_gt_0_1']}
        metrics = {}
        for row in rows:
            seed = int(row['seed'])
            if seed not in included:
                continue
            key = seed, row['mask'], int(row['layer']), row['target']
            metrics.setdefault(key, []).append(float(row['similarity']))
        rng = np.random.default_rng(20261127)
        out = []
        for layer in range(5):
            for target in LABELS:
                paired = [s for s in sorted(included) if all((s, m, layer, target) in metrics for m in cfg['masks'])]
                if not paired:
                    continue
                delta = np.array([np.mean(metrics[s, '111', layer, target]) - np.mean(metrics[s, '000', layer, target]) for s in paired])
                boot = delta[rng.integers(len(delta), size=(20000, len(delta)))].mean(1)
                lo, hi = np.quantile(boot, [.025, .975])
                for mask in cfg['masks']:
                    vals = [np.mean(metrics[s, mask, layer, target]) for s in paired]
                    out.append({'layer': layer, 'target': target, 'mask': mask, 'model_seeds': len(vals),
                        'mean_similarity': float(np.mean(vals)), 'mean_error': float(1 - np.mean(vals)),
                        'sd_across_model_seeds': float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.,
                        'paired_on_minus_off': float(delta.mean()), 'paired_ci_low': float(lo), 'paired_ci_high': float(hi)})
        write_csv(args.output / ('matched-summary.csv' if matched else 'summary.csv'), out)

    def fit_targets(features, targets, output, seed, mask):
        output.mkdir(parents=True, exist_ok=True)
        rows, cross_rows = [], []
        for split in cfg['probe_split_seeds']:
            fit, validation, test = dp.partition(features.shape[1], split)
            fit_seed = 20261125 + 31 * seed + split
            fitted = fit_maps_trace(features, targets, fit, validation, **settings, seed=fit_seed, device=device)
            values, cross = dp.score_maps(features, targets, fitted, test)
            shuffled_targets = targets.copy()
            rng = np.random.default_rng(fit_seed + 1)
            for indices in (fit, validation):
                shuffled_targets[:, indices] = targets[:, rng.permutation(indices)]
            negative_fit = fit_maps_trace(features, shuffled_targets, fit, validation,
                **settings, seed=fit_seed, device=device, restarts=('random',))
            negative, _ = dp.score_maps(features, targets, negative_fit, test)
            fixed = normalize(features[:, test, cfg['d']:])
            fixed_values = np.einsum('lnd,tnd->ltn', fixed, targets[:, test]).clip(-1, 1) ** 2
            for layer in range(5):
                for index, label in enumerate(LABELS):
                    similarity = float(values[layer, index].mean())
                    rows.append({'seed': seed, 'mask': mask, 'layer': layer, 'target': label,
                        'power': POWERS[index], 'split_seed': split,
                        'fit_samples': len(fit), 'validation_samples': len(validation), 'test_samples': len(test),
                        'similarity': similarity, 'mean_error': 1 - similarity,
                        'corrected_similarity': (similarity - 1 / cfg['d']) / (1 - 1 / cfg['d']),
                        'fixed_similarity': float(fixed_values[layer, index].mean()),
                        'shuffled_similarity': float(negative[layer, index].mean()),
                        'validation_similarity': float(fitted['validation_similarity'][layer, index]),
                        'best_step': int(fitted['best_step'][layer, index]), 'restart': str(fitted['restart'][layer, index])})
                    for other, scored_label in enumerate(LABELS):
                        cross_rows.append({'seed': seed, 'mask': mask, 'split_seed': split, 'layer': layer,
                            'fitted_target': label, 'scored_target': scored_label,
                            'similarity': float(cross[layer, index, other])})
            np.savez(output / f'split-{split}.npz', **fitted, fit_indices=fit,
                validation_indices=validation, test_indices=test, targets=np.asarray(LABELS))
        write_csv(output / 'scores.csv', rows)
        write_csv(output / 'cross-target-scores.csv', cross_rows)
        atomic_json(output / 'complete.json', {'scores': len(rows), 'targets': LABELS})
        return rows

    all_rows, audits = [], []
    target_dir = args.output / 'target-data'
    target_dir.mkdir(exist_ok=True)
    for seed in cfg['seeds']:
        record = json.loads((args.parent / f'seed-{seed}-data.json').read_text())
        with np.load(args.probe_data / f'seed-{seed}.npz') as saved:
            inputs = saved['inputs']
            old_directions = saved['power_directions']
            eigenvector = normalize(saved['eigenvector_reference'])
        assert digest(inputs) == record['splits']['probe']['input_sha256'], f'Input mismatch: {seed}'
        assert digest(old_directions) == record['power_sha256'], f'Parent labels mismatch: {seed}'
        directions = power_directions(inputs, max(POWERS))
        assert digest(directions[:, :17]) == record['power_sha256'], f'Iteration prefix mismatch: {seed}'
        targets = np.stack([directions[:, k] for k in POWERS], axis=0)
        np.savez_compressed(target_dir / f'seed-{seed}.npz', targets=targets, labels=np.asarray(LABELS),
            parent_input_sha256=np.asarray(record['splits']['probe']['input_sha256']))
        alignment = {f'q{k}_to_eigenvector': float(np.mean(np.einsum('nd,nd->n', directions[:, k], eigenvector).clip(-1, 1) ** 2)) for k in [16, *POWERS]}
        audit = {'seed': seed, 'inputs_verified': True, 'parent_labels_verified': True,
                 'power_prefix_verified': True, 'target_alignment': alignment, 'arms': {}}
        for mask in cfg['masks']:
            folder = args.output / f'seed-{seed}' / mask
            checkpoint = args.parent / f'seed-{seed}' / mask / 'selected.pt'
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            if (folder / 'complete.json').exists():
                with (folder / 'scores.csv').open() as handle:
                    all_rows.extend(list(csv.DictReader(handle)))
                audit['arms'][mask] = {'checkpoint_sha256': before, 'reused_completed_probes': True}
                continue
            model = model_for(mask, cfg['init_std'], device, cfg['precision'])
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval().requires_grad_(False)
            features, fixed = collect_query_states(model, inputs, cfg['batch_size'], device)
            baseline_differences = []
            for split in cfg['probe_split_seeds']:
                _, _, test = dp.partition(len(inputs), split)
                baseline = float(np.mean(np.einsum('nd,nd->n', fixed[4, test], eigenvector[test]).clip(-1, 1) ** 2))
                baseline_differences.append(abs(baseline - prior_eigen_scores[seed, mask, split]))
            assert max(baseline_differences) < 1e-10, f'True-target baseline mismatch: {seed}/{mask}'
            rows = fit_targets(features, targets, folder, seed, mask)
            all_rows.extend(rows)
            assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
            audit['arms'][mask] = {'checkpoint_sha256': before, 'checkpoint_unchanged': True,
                'maximum_true_eigen_baseline_difference': max(baseline_differences)}
            write_csv(args.output / 'scores.csv', all_rows)
            print(f'SUPPLEMENT_ARM_COMPLETE seed={seed} mask={mask} scores={len(rows)}', flush=True)
            del model, features, fixed
            torch.cuda.empty_cache()
        audits.append(audit)
        atomic_json(args.output / 'data-and-model-audit.json', audits)
        summarize(all_rows)
        summarize(all_rows, matched=True)
        print(f'SUPPLEMENT_PAIR_COMPLETE seed={seed}', flush=True)
        del inputs, old_directions, directions, targets, eigenvector
    atomic_json(args.output / 'complete.json', {'status': 'complete', 'pairs': len(cfg['seeds']),
        'models': 2 * len(cfg['seeds']), 'scores': len(all_rows), 'targets': LABELS,
        'probe_steps': settings['steps'], 'mother_models_retrained': False})
    print('SUPPLEMENT_STUDY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
