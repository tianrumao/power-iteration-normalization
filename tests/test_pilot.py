import io
import unittest

import numpy as np
import torch

from src.data import sample
from src.model import Model, direction_error, readout


class PilotChecks(unittest.TestCase):
    def test_eigenvectors_and_power_steps(self):
        data = sample(8, 4, 7, np.random.default_rng(8))
        x = data.tokens[:, :4, :7]
        np.testing.assert_allclose(np.linalg.norm(x, axis=1), 1)
        np.testing.assert_allclose(np.linalg.norm(data.power, axis=-1), 1)
        np.testing.assert_allclose(data.tokens[:, 4:, -1], data.power[:, 0])
        s = x @ x.swapaxes(-1, -2)
        top = np.linalg.eigvalsh(s)[:, -1, None] * data.target
        np.testing.assert_allclose((s @ data.target[..., None])[..., 0], top, atol=1e-12)
        update = (s @ data.power[:, 0, :, None])[..., 0]
        np.testing.assert_allclose(data.power[:, 1], update / np.linalg.norm(update, axis=-1, keepdims=True))

    def test_sign_and_invalid_predictions(self):
        tokens = torch.zeros(4, 4, 2)
        tokens[:, 2, -1] = torch.tensor([1.0, -1.0, 0.0, float("nan")])
        pred, valid = readout(tokens)
        target = torch.tensor([[1.0, 0.0]]).expand_as(pred)
        torch.testing.assert_close(direction_error(pred, target, valid), torch.tensor([0., 0., 1., 1.]))
        self.assertEqual(valid.tolist(), [True, True, False, False])

    def test_block_bypass_and_reload(self):
        x = torch.randn(3, 4, 5)
        model = Model(2, True, 0.08)
        block = model.blocks[0]
        torch.testing.assert_close(block(x), x + block.v @ x @ x.transpose(-1, -2) @ block.w @ x)
        intact, _ = model(x)
        bypass, _ = model(x, bypass=0)
        torch.testing.assert_close(bypass[0], x / x.norm(dim=1, keepdim=True))
        matched, possible = model(x, bypass=3, reference=intact[3])
        self.assertTrue(possible.all())
        torch.testing.assert_close(matched[3].norm(dim=1), intact[3].norm(dim=1))
        saved = io.BytesIO()
        torch.save(model.state_dict(), saved)
        saved.seek(0)
        restored = Model(2, True, 0.08)
        restored.load_state_dict(torch.load(saved, weights_only=True))
        for options in ({}, {"bypass": 1}, {"bypass": 3, "reference": intact[3]}):
            original, _ = model(x, **options)
            reloaded, _ = restored(x, **options)
            for lhs, rhs in zip(original, reloaded):
                torch.testing.assert_close(lhs, rhs)
