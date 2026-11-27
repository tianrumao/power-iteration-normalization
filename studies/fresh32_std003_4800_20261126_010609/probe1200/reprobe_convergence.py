"""Refit every direction probe with a longer budget, preserving the parent study."""
import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np
import torch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--parent', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1200)
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

    assert torch.cuda.is_available(), 'Select a GPU runtime.'
    device = torch.device('cuda')
    torch.set_num_threads(1)
    cfg = json.loads((args.parent / 'config.json').read_text())
    cfg['probe_steps'] = args.steps
    powers = json.loads((args.parent / 'power-selection.json').read_text())['powers']
    args.output.mkdir(parents=True, exist_ok=True)
    atomic_json(args.output / 'config.json', cfg)
    atomic_json(args.output / 'protocol.json', {
        'parent_run': args.parent.name, 'mother_models_retrained': False,
        'original_probe_steps': 300, 'probe_steps': args.steps,
        'selection': 'validation only, including fixed readout at step zero',
        'data': 'Regenerated probe pool only; require original input and power-label SHA256.',
        'model_source': 'Original selected.pt, no mother-model parameter updates',
        'versions': {'python': sys.version, 'numpy': np.__version__, 'torch': str(torch.__version__)},
    })

    def fit_maps_trace(features, targets, fit, validation, *, steps=1200,
                       learning_rate=.01, batch_size=256, seed=0,
                       device='cuda', restarts=('fixed', 'random')):
        layers, _, width = features.shape
        count, _, d = targets.shape
        features = normalize(features)
        rms = np.sqrt(np.mean(features[:, fit] ** 2, axis=1, keepdims=True))
        rms = np.where(rms < 1e-12, 1, rms)
        z = torch.as_tensor(features / rms, dtype=torch.float64, device=device)
        y = torch.as_tensor(normalize(targets), dtype=torch.float64, device=device)
        initial = np.zeros((layers, count, d, width))
        for layer in range(layers):
            initial[layer, :, np.arange(d), d + np.arange(d)] = rms[layer, 0, d:][:, None]
        fixed = torch.as_tensor(initial, dtype=z.dtype, device=device)
        best_weights = fixed.clone()
        best_value = torch.full((layers, count), -1., dtype=z.dtype, device=device)
        best_step = np.zeros((layers, count), dtype=int)
        best_restart = np.full((layers, count), 'fixed', dtype='U8')
        generator = torch.Generator(device=device).manual_seed(seed)
        history_values, history_steps, history_restart = [], [], []
        for restart_index, restart in enumerate(restarts):
            weights = torch.nn.Parameter(fixed.clone() if restart == 'fixed' else
                torch.randn(initial.shape, dtype=z.dtype, device=device, generator=generator) / np.sqrt(width))
            optimizer = torch.optim.Adam([weights], lr=learning_rate)
            rng = np.random.default_rng(seed)
            for step in range(steps + 1):
                if step % 25 == 0 or step == steps:
                    with torch.no_grad():
                        prediction = torch.einsum('lnf,ltdf->ltnd', z[:, validation], weights)
                        current = dp.similarity(prediction, y[:, validation][None]).mean(-1)
                        if not torch.isfinite(current).all():
                            raise FloatingPointError('Nonfinite validation score')
                        improved = current > best_value
                        best_value[improved] = current[improved]
                        best_weights[improved] = weights.detach()[improved]
                        changed = improved.cpu().numpy()
                        best_step[changed] = step
                        best_restart[changed] = restart
                        history_values.append(current.cpu().numpy())
                        history_steps.append(step)
                        history_restart.append(restart_index)
                if step == steps:
                    break
                indices = fit[rng.integers(0, len(fit), batch_size)]
                optimizer.zero_grad(set_to_none=True)
                prediction = torch.einsum('lnf,ltdf->ltnd', z[:, indices], weights)
                loss = (1 - dp.similarity(prediction, y[:, indices][None])).mean(-1).sum()
                loss.backward()
                if not torch.isfinite(loss) or not torch.isfinite(weights.grad).all():
                    raise FloatingPointError('Nonfinite probe update')
                optimizer.step()
        return dict(weights=best_weights.cpu().numpy(), rms=rms,
                    validation_similarity=best_value.cpu().numpy(),
                    best_step=best_step, restart=best_restart,
                    history_similarity=np.asarray(history_values),
                    history_steps=np.asarray(history_steps),
                    history_restart=np.asarray(history_restart))

    # The original scorer, partitions, losses, negative controls and CSV schema stay fixed.
    dp.fit_maps = fit_maps_trace
    all_rows, audits = [], []
    for seed in cfg['seeds']:
        manifest = json.loads((args.parent / f'seed-{seed}-data.json').read_text())
        stream = np.random.SeedSequence([cfg['data_epoch'], seed, 10723]).spawn(4)[3]
        generated = sample(cfg['probe_samples'], cfg['d'], cfg['n'], np.random.default_rng(stream), power_steps=0)
        inputs = generated.tokens.astype(np.float32)
        directions = power_directions(inputs, cfg['max_power'])
        assert digest(inputs) == manifest['splits']['probe']['input_sha256'], f'Probe input mismatch: {seed}'
        assert digest(directions) == manifest['power_sha256'], f'Power-label mismatch: {seed}'
        audit = {'seed': seed, 'probe_input_matches': True, 'power_labels_match': True, 'arms': {}}
        for mask in cfg['masks']:
            folder = args.output / f'seed-{seed}' / mask
            scores_path = folder / 'scores.csv'
            if (folder / 'complete.json').exists():
                with scores_path.open() as handle:
                    all_rows.extend(list(csv.DictReader(handle)))
                continue
            checkpoint = args.parent / f'seed-{seed}' / mask / 'selected.pt'
            before = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
            model = model_for(mask, cfg['init_std'], device, cfg['precision'])
            model.load_state_dict(torch.load(checkpoint, map_location=device, weights_only=True))
            model.eval().requires_grad_(False)
            features, fixed = collect_query_states(model, inputs, cfg['batch_size'], device)
            with (args.parent / f'seed-{seed}' / mask / 'power_probes/scores.csv').open() as handle:
                old = list(csv.DictReader(handle))
            fixed_diff = []
            for split_seed in cfg['probe_split_seeds']:
                _, _, test = dp.partition(len(inputs), split_seed)
                original = next(r for r in old if int(r['layer']) == 4 and int(r['power']) == 16 and int(r['split_seed']) == split_seed)
                value = np.mean(np.einsum('nd,nd->n', fixed[4, test], directions[test, 16]).clip(-1, 1) ** 2)
                fixed_diff.append(abs(value - float(original['fixed_similarity'])))
            assert max(fixed_diff) < 1e-10, f'Original final-layer readout mismatch: {seed}/{mask}'
            rows = dp.probe_powers(features, directions, powers, cfg, folder,
                                   {'seed': seed, 'mask': mask}, device)
            all_rows.extend(rows)
            same = hashlib.sha256(checkpoint.read_bytes()).hexdigest() == before
            assert same
            audit['arms'][mask] = {'checkpoint_sha256': before, 'checkpoint_unchanged': same,
                                  'maximum_original_readout_difference': max(fixed_diff)}
            write_csv(args.output / 'power-probes.csv', all_rows)
            print(f'PROBE_ARM_COMPLETE seed={seed} mask={mask} rows={len(rows)}', flush=True)
            del model, features, fixed
            torch.cuda.empty_cache()
        audits.append(audit)
        atomic_json(args.output / 'data-and-model-audit.json', audits)
        del generated, inputs, directions
        print(f'PROBE_PAIR_COMPLETE seed={seed}', flush=True)
    write_csv(args.output / 'power-probes.csv', all_rows)
    atomic_json(args.output / 'complete.json', {'status': 'complete', 'pairs': len(cfg['seeds']),
                'models': len(cfg['seeds']) * 2, 'probe_scores': len(all_rows), 'steps': args.steps,
                'mother_models_retrained': False})
    print('PROBE_STUDY_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
