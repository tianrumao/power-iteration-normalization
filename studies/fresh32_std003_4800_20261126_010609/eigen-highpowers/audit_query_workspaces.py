"""Read-only upper/lower query-direction audit on the exact probe test examples."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
import torch


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--parent', type=Path, required=True)
    p.add_argument('--probe-data', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--source', type=Path, required=True)
    args = p.parse_args()
    sys.path.insert(0, str(args.source))
    from src.training import model_for
    from src.probe_math import collect_query_states
    from src.power_targets import normalize
    from src.direction_probes import partition
    from src.io import write_csv, atomic_json, digest
    cfg = json.loads((args.parent / 'config.json').read_text())
    torch.set_num_threads(1)
    rows, audits = [], []
    for seed in cfg['seeds']:
        with np.load(args.probe_data / f'seed-{seed}.npz') as saved:
            x = saved['inputs']
            truth = normalize(saved['eigenvector_reference'])
        expected = json.loads((args.parent / f'seed-{seed}-data.json').read_text())
        assert digest(x) == expected['splits']['probe']['input_sha256']
        for mask in cfg['masks']:
            path = args.parent / f'seed-{seed}' / mask / 'selected.pt'
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            model = model_for(mask, cfg['init_std'], torch.device('cuda'), cfg['precision'])
            model.load_state_dict(torch.load(path, map_location='cuda', weights_only=True))
            model.eval().requires_grad_(False)
            features, _ = collect_query_states(model, x, cfg['batch_size'], torch.device('cuda'))
            for workspace, z in [('upper', features[:, :, :cfg['d']]), ('lower', features[:, :, cfg['d']:])]:
                valid = np.max(np.abs(z), axis=-1) > 0
                direction = normalize(z)
                values = np.einsum('lnd,nd->ln', direction, truth).clip(-1, 1) ** 2
                for split in cfg['probe_split_seeds']:
                    _, _, test = partition(len(x), split)
                    for layer in range(5):
                        okay = valid[layer, test]
                        rows.append({'seed': seed, 'mask': mask, 'layer': layer, 'workspace': workspace,
                            'split_seed': split, 'samples': len(test), 'valid_directions': int(okay.sum()),
                            'similarity_zero_for_invalid': float(values[layer, test].mean()),
                            'similarity_valid_only': float(values[layer, test][okay].mean()) if okay.any() else ''})
            assert hashlib.sha256(path.read_bytes()).hexdigest() == before
            audits.append({'seed': seed, 'mask': mask, 'checkpoint_unchanged': True, 'checkpoint_sha256': before})
            del model, features
            torch.cuda.empty_cache()
        write_csv(args.output / 'query-workspaces.csv', rows)
        print(f'WORKSPACE_AUDIT seed={seed}', flush=True)
    atomic_json(args.output / 'query-workspaces-complete.json', {'status': 'complete', 'rows': len(rows),
        'models': len(audits), 'label': 'true eigenvector', 'training_performed': False,
        'note': 'Fixed coordinate readouts, not newly trained probes; initial upper query is zero and has no direction.',
        'audits': audits})
    print('WORKSPACE_AUDIT_COMPLETE', flush=True)


if __name__ == '__main__':
    main()
