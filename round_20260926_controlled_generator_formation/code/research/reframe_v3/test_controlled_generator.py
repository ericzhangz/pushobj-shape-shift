"""CPU-only invariants for the standard controlled-generator reference."""

import io
import math
import unittest

import torch
from torch import nn

from research.reframe_v3.controlled_generator import ControlledGenerator


def _history(batch=2, length=3, obs_dim=2, dtype=torch.float64):
    return torch.arange(batch * length * (obs_dim + 10), dtype=dtype).reshape(
        batch, length, 1, obs_dim + 10) / 20.


class PolynomialFields(nn.Module):
    """X=partial_x, Y=x^2 partial_y: an explicit order-sensitive toy, not physics."""

    def forward(self, inputs):
        x = inputs[:, -2]
        zero = torch.zeros_like(x)
        one = torch.ones_like(x)
        return torch.stack((zero, zero, one, zero, zero, x.square()), dim=-1)


class ExponentialDrift(nn.Module):
    def forward(self, inputs):
        x = inputs[:, -2]
        zero = torch.zeros_like(x)
        return torch.stack((x, zero, zero, zero, zero, zero), dim=-1)


class ControlledGeneratorTests(unittest.TestCase):
    def _nonzero(self, kind="flow"):
        torch.manual_seed(41)
        model = ControlledGenerator(obs_dim=2, hidden=12, kind=kind).double()
        network = model.fields if kind == "flow" else model.packed_net
        with torch.no_grad():
            network[-1].weight.normal_(0., .08)
            network[-1].bias.normal_(0., .02)
        return model

    def test_zero_initialization_is_exact_identity_not_f0(self):
        history = _history()
        actions = torch.linspace(-.7, .8, 20, dtype=torch.float64).reshape(2, 1, 10)
        for kind in ("flow", "packed"):
            model = ControlledGenerator(obs_dim=2, hidden=12, kind=kind).double()
            result = model(history, actions)
            self.assertEqual(result.shape, (2, 1, 1, 2))
            self.assertTrue(torch.equal(result[:, 0, 0], history[:, -1, 0, :2]))
            if kind == "flow":
                self.assertTrue(torch.equal(result, model(history, actions, substeps=2)))

    def test_context_preserves_past_tokens_masks_and_native_truncation(self):
        model = ControlledGenerator(obs_dim=2, hidden=8).double()
        model.configure_normalization(torch.tensor([1., 2.]), torch.tensor([2., 4.]))
        history = _history(batch=1, length=4)
        context = model.build_context(history).reshape(1, 2, 13)
        expected = history[:, 1:3, 0].clone()
        expected[..., :2] = (expected[..., :2] - model.observation_center) / model.observation_scale
        self.assertTrue(torch.equal(context[..., :12], expected))
        self.assertTrue(torch.equal(context[..., 12], torch.ones(1, 2, dtype=torch.float64)))
        short = model.build_context(history[:, :2]).reshape(1, 2, 13)
        self.assertTrue(torch.equal(short[:, 0], torch.zeros(1, 13, dtype=torch.float64)))
        self.assertEqual(short[0, 1, -1].item(), 1.)
        lone = model.build_context(history[:, :1])
        self.assertTrue(torch.equal(lone, torch.zeros_like(lone)))
        no_past = ControlledGenerator(obs_dim=2, num_hist=1, hidden=8).double()
        self.assertEqual(no_past.build_context(history).shape, (1, 0))

    def test_current_encoded_action_does_not_enter_context_or_state(self):
        history = _history(batch=1)
        changed = history.clone()
        changed[:, -1, 0, 2:] += 700.
        raw = torch.linspace(-.2, .4, 10, dtype=torch.float64).reshape(1, 1, 10)
        for kind in ("flow", "packed"):
            model = self._nonzero(kind)
            self.assertTrue(torch.equal(model.build_context(history), model.build_context(changed)))
            self.assertTrue(torch.equal(model(history, raw), model(changed, raw)))
        changed[:, -2, 0, 2:] += 1.
        self.assertFalse(torch.equal(model.build_context(history), model.build_context(changed)))

    def test_raw_action_only_enters_shared_flow_through_current_two_command(self):
        model = ControlledGenerator(obs_dim=2, hidden=8).double()
        model.fields = PolynomialFields()
        history = torch.zeros(1, 1, 1, 12, dtype=torch.float64)
        a, b, zero = [1., 0.], [0., 1.], [0., 0.]
        abba = torch.tensor([[a, b, b, a, zero]], dtype=torch.float64).reshape(1, 1, 10)
        baab = torch.tensor([[b, a, a, b, zero]], dtype=torch.float64).reshape(1, 1, 10)
        self.assertTrue(torch.equal(abba.reshape(1, 5, 2).sum(1),
                                    baab.reshape(1, 5, 2).sum(1)))
        first, second = model(history, abba), model(history, baab)
        torch.testing.assert_close(first[0, 0, 0], torch.tensor([.4, .016], dtype=torch.float64))
        torch.testing.assert_close(second[0, 0, 0], torch.tensor([.4, .032], dtype=torch.float64))
        self.assertFalse(torch.equal(first, second))
        # Numerical refinement uses these exact same fields, not a second model.
        torch.testing.assert_close(first, model(history, abba, substeps=2))

    def test_substep_refinement_improves_same_field_rk2_error(self):
        model = ControlledGenerator(obs_dim=2, hidden=8).double()
        model.fields = ExponentialDrift()
        history = torch.zeros(1, 1, 1, 12, dtype=torch.float64)
        history[..., 0] = 1.
        actions = torch.zeros(1, 1, 10, dtype=torch.float64)
        coarse = model(history, actions)[0, 0, 0, 0].item()
        refined = model(history, actions, substeps=2)[0, 0, 0, 0].item()
        self.assertAlmostEqual(coarse, (1. + .2 + .2**2 / 2.)**5, places=13)
        self.assertAlmostEqual(refined, (1. + .1 + .1**2 / 2.)**10, places=13)
        self.assertLess(abs(refined - math.e), abs(coarse - math.e))

    def test_action_gradient_matches_finite_difference(self):
        history = _history(batch=1)
        for kind in ("flow", "packed"):
            model = self._nonzero(kind)
            actions = torch.linspace(-.3, .4, 10, dtype=torch.float64).reshape(1, 1, 10)
            actions.requires_grad_(True)
            value = model(history, actions).sum()
            gradient, = torch.autograd.grad(value, actions)
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().max().item(), 1e-7)
            eps = 1e-6
            for coordinate in (0, 3, 9):
                plus, minus = actions.detach().clone(), actions.detach().clone()
                plus[0, 0, coordinate] += eps
                minus[0, 0, coordinate] -= eps
                numerical = ((model(history, plus).sum() - model(history, minus).sum())
                             / (2 * eps))
                torch.testing.assert_close(gradient[0, 0, coordinate], numerical,
                                           atol=2e-8, rtol=2e-6)

    def test_state_dict_roundtrip_restores_exact_predictions_and_scaling(self):
        history = _history(batch=1)
        actions = torch.linspace(-.2, .4, 10, dtype=torch.float64).reshape(1, 1, 10)
        for kind in ("flow", "packed"):
            model = self._nonzero(kind)
            model.configure_normalization(torch.tensor([.2, -.3]), torch.tensor([.7, 1.2]))
            buffer = io.BytesIO()
            torch.save(model.state_dict(), buffer)
            buffer.seek(0)
            restored = ControlledGenerator(obs_dim=2, hidden=12, kind=kind).double()
            restored.load_state_dict(torch.load(buffer, weights_only=True))
            self.assertTrue(torch.equal(model(history, actions), restored(history, actions)))
            if kind == "flow":
                self.assertTrue(torch.equal(model(history, actions, substeps=2),
                                            restored(history, actions, substeps=2)))

    def test_scale_floor_is_explicit_fixed_and_native_unit_output(self):
        model = ControlledGenerator(obs_dim=2, hidden=8).double()
        model.configure_normalization(torch.tensor([2., -1.]), torch.tensor([0., 1e-5]))
        torch.testing.assert_close(model.observation_scale, torch.full((2,), 1e-3, dtype=torch.float64))
        history = _history(batch=1)
        scale_before = model.observation_scale.clone()
        center_before = model.observation_center.clone()
        result = model(history, torch.zeros(1, 1, 10, dtype=torch.float64))
        self.assertTrue(torch.equal(result[:, 0, 0], history[:, -1, 0, :2]))
        self.assertTrue(torch.equal(scale_before, model.observation_scale))
        self.assertTrue(torch.equal(center_before, model.observation_center))
        model.configure_normalization(torch.zeros(2), torch.tensor([2., 3.]))
        with torch.no_grad():
            model.fields[-1].bias[0] = 1.
        start = torch.zeros(1, 1, 1, 12, dtype=torch.float64)
        output = model(start, torch.zeros(1, 1, 10, dtype=torch.float64))
        torch.testing.assert_close(output[0, 0, 0], torch.tensor([2., 0.], dtype=torch.float64))

    def test_parameter_budget_is_reported_not_claimed_matched(self):
        flow = ControlledGenerator()
        packed = ControlledGenerator(kind="packed")
        self.assertEqual(flow.parameter_count(), 323230)
        self.assertEqual(packed.parameter_count(), 222858)
        self.assertEqual(flow.parameter_count(), sum(p.numel() for p in flow.parameters()))
        self.assertNotEqual(flow.parameter_count(), packed.parameter_count())

    def test_invalid_inputs_fail_explicitly(self):
        for kwargs in ({"kind": "unknown"}, {"obs_dim": 0}, {"num_hist": 0},
                       {"hidden": True}, {"action_emb_dim": -1}):
            with self.assertRaises(ValueError):
                ControlledGenerator(**kwargs)
        model = ControlledGenerator(obs_dim=2, hidden=8).double()
        history = _history(batch=1)
        action = torch.zeros(1, 1, 10, dtype=torch.float64)
        for malformed in (history[:, :0], history.squeeze(2), history[..., :-1],
                          history.expand(1, 3, 2, 12), history.float(),
                          torch.full_like(history, float("nan"))):
            with self.assertRaises(ValueError):
                model(malformed, action)
        for malformed in (action[:, 0], action[..., :9], action.expand(2, 1, 10),
                          action.float(), torch.full_like(action, float("inf"))):
            with self.assertRaises(ValueError):
                model(history, malformed)
        for substeps in (0, 3, True, 1.0):
            with self.assertRaises(ValueError):
                model(history, action, substeps=substeps)
        with self.assertRaises(ValueError):
            ControlledGenerator(obs_dim=2, hidden=8, kind="packed").double()(
                history, action, substeps=2)
        for center, scale in ((torch.zeros(3), torch.ones(2)),
                              (torch.zeros(2), -torch.ones(2)),
                              (torch.full((2,), float("nan")), torch.ones(2))):
            with self.assertRaises(ValueError):
                model.configure_normalization(center, scale)
        with torch.no_grad():
            model.fields[-1].bias[0] = float("nan")
        with self.assertRaises(FloatingPointError):
            model(history, action)


if __name__ == "__main__":
    unittest.main()
