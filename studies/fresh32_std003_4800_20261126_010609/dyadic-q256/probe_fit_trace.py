"""Same validation-selected fitter used in the 1,200-step probe audit."""
import numpy as np
import torch
from src.power_targets import normalize
from src import direction_probes as dp

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
