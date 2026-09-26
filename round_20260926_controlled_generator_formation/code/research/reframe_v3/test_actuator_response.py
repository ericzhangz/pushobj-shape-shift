"""Synthetic CPU tests for the explicitly bounded actuator hypothesis."""

from types import SimpleNamespace
import unittest

import torch

from research.reframe_v3.actuator_response import fit_actuator_response, make_actuator_transition


def _preprocessor():
    return SimpleNamespace(
        proprio_mean=torch.tensor([4., -1., .3, -.7], dtype=torch.float64),
        proprio_std=torch.tensor([2., 3., .5, .9], dtype=torch.float64),
        action_mean=torch.tensor([.25, -.1], dtype=torch.float64),
        action_std=torch.tensor([.7, 1.3], dtype=torch.float64))


def _matrix():
    return torch.tensor([[.4, 0., .7, 0.], [0., .3, 0., .6],
                         [.2, .05, .25, .03], [-.04, .1, -.02, .35]], dtype=torch.float64)


def _chain(matrix=None, length=20):
    matrix = _matrix() if matrix is None else matrix
    generator = torch.Generator().manual_seed(3701)
    actions = torch.randn(length, 2, generator=generator, dtype=torch.float64)
    observations = [torch.tensor([2., -3., .8, -.5], dtype=torch.float64)]
    for action in actions:
        previous = observations[-1]
        response = torch.cat((previous[2:], action)) @ matrix
        observations.append(torch.cat((previous[:2] + response[:2], response[2:])))
    preprocessor = _preprocessor()
    normalized_prop = (torch.stack(observations)[None] - preprocessor.proprio_mean) / preprocessor.proprio_std
    normalized_action = (actions[None] - preprocessor.action_mean) / preprocessor.action_std
    return normalized_prop, normalized_action, preprocessor


class ActuatorResponseTests(unittest.TestCase):
    def test_exact_linear_identification_and_normalization_are_saved(self):
        observed, actions, preprocessor = _chain()
        fitted = fit_actuator_response(observed, actions, preprocessor)
        torch.testing.assert_close(fitted["M"], _matrix(), rtol=1e-12, atol=1e-12)
        self.assertEqual(fitted["rank"], 4)
        self.assertEqual(fitted["num_transitions"], 20)
        self.assertEqual(fitted["M"].dtype, torch.float64)
        self.assertLess(fitted["fit_max_abs"], 1e-12)
        for key in ("proprio_mean", "proprio_std", "action_mean", "action_std"):
            self.assertTrue(torch.equal(fitted[key], getattr(preprocessor, key)))

    def test_five_step_composition_matches_known_accumulated_motion(self):
        matrix = torch.tensor([[1., 0., 1., 0.], [0., 1., 0., 1.],
                               [1., 0., 1., 0.], [0., 1., 0., 1.]], dtype=torch.float64)
        observed, actions, preprocessor = _chain(matrix)
        transition = make_actuator_transition(fit_actuator_response(observed, actions, preprocessor))
        state = torch.tensor([[[2., -3., .5, -.25]]], dtype=torch.float64)
        raw_actions = torch.tensor([[[1., 0.], [0., 1.], [-.5, .5], [.25, -.5], [0., -1.]]],
                                   dtype=torch.float64)
        state = (state - preprocessor.proprio_mean) / preprocessor.proprio_std
        chunk = ((raw_actions - preprocessor.action_mean) / preprocessor.action_std).reshape(1, 1, 10)
        result = transition(state, chunk) * preprocessor.proprio_std + preprocessor.proprio_mean
        expected = torch.tensor([[[8.5, -.75, 1.25, -.25]]], dtype=torch.float64)
        torch.testing.assert_close(result, expected, rtol=1e-12, atol=1e-12)

    def test_translation_equivariance_and_batch_behavior(self):
        observed, actions, preprocessor = _chain()
        transition = make_actuator_transition(fit_actuator_response(observed, actions, preprocessor))
        shift = torch.tensor([7., -11., 0., 0.], dtype=torch.float64)
        state = observed[:, :1]
        translated = state + shift / preprocessor.proprio_std
        chunk = actions[:, :5].reshape(1, 1, 10)
        result = transition(torch.cat((state, translated)), chunk.repeat(2, 1, 1))
        torch.testing.assert_close((result[1] - result[0]) * preprocessor.proprio_std,
                                  shift[None], rtol=1e-12, atol=1e-12)
        shifted_observations = observed + shift / preprocessor.proprio_std
        shifted_fit = fit_actuator_response(shifted_observations, actions, preprocessor)
        torch.testing.assert_close(shifted_fit["M"], _matrix(), rtol=1e-12, atol=1e-12)

    def test_rank_deficiency_and_extra_future_observations_fail(self):
        observed, actions, preprocessor = _chain()
        raw_zero_prop = torch.zeros_like(observed)
        raw_zero_actions = torch.zeros_like(actions)
        normalized_zero_prop = (raw_zero_prop - preprocessor.proprio_mean) / preprocessor.proprio_std
        normalized_zero_actions = (raw_zero_actions - preprocessor.action_mean) / preprocessor.action_std
        with self.assertRaisesRegex(ValueError, "rank four"):
            fit_actuator_response(normalized_zero_prop, normalized_zero_actions, preprocessor)
        with self.assertRaisesRegex(ValueError, "exactly N\\+1"):
            fit_actuator_response(observed, actions[:, :10], preprocessor)
        transition = make_actuator_transition(fit_actuator_response(observed, actions, preprocessor))
        with self.assertRaisesRegex(ValueError, "batch,1,10"):
            transition(observed[:, :1], actions[:, :4].reshape(1, 1, 8))

    def test_autograd_matches_finite_differences(self):
        observed, actions, preprocessor = _chain()
        transition = make_actuator_transition(fit_actuator_response(observed, actions, preprocessor))
        state = observed[:, :1].clone().requires_grad_(True)
        chunk = actions[:, :5].reshape(1, 1, 10).clone().requires_grad_(True)
        weights = torch.tensor([.2, -.3, .7, .1], dtype=torch.float64)
        output = (transition(state, chunk) * weights).sum()
        state_grad, action_grad = torch.autograd.grad(output, (state, chunk))
        epsilon = 1e-6
        for variable, derivative, positions in ((state, state_grad, (0, 2, 3)),
                                                (chunk, action_grad, (0, 5, 9))):
            for position in positions:
                plus, minus = variable.detach().clone(), variable.detach().clone()
                plus[0, 0, position] += epsilon
                minus[0, 0, position] -= epsilon
                if variable is state:
                    difference = transition(plus, chunk.detach()) - transition(minus, chunk.detach())
                else:
                    difference = transition(state.detach(), plus) - transition(state.detach(), minus)
                finite = float((difference * weights).sum() / (2 * epsilon))
                self.assertAlmostEqual(float(derivative[0, 0, position]), finite, places=8)

    def test_optional_path_observer_uses_exact_same_recurrence(self):
        observed, actions, preprocessor = _chain()
        fitted = fit_actuator_response(observed, actions, preprocessor)
        paths = []
        traced = make_actuator_transition(fitted, path_observer=paths.append)
        plain = make_actuator_transition(fitted)
        state, chunk = observed[:, :1], actions[:, :5].reshape(1, 1, 10)
        self.assertTrue(torch.equal(traced(state, chunk), plain(state, chunk)))
        self.assertEqual(paths[0].shape, (1, 6, 4))
        actual_raw = observed[:, :6] * preprocessor.proprio_std + preprocessor.proprio_mean
        torch.testing.assert_close(paths[0], actual_raw, atol=1e-12, rtol=1e-12)
        self.assertTrue(torch.equal(plain(state, chunk),
            ((paths[0][:, -1:] - preprocessor.proprio_mean) / preprocessor.proprio_std)))
        with self.assertRaisesRegex(ValueError, "callable"):
            make_actuator_transition(fitted, path_observer=4)

    def test_prefix_fit_is_independent_of_unpassed_future_storage(self):
        observed, actions, preprocessor = _chain()
        prefix_observed, prefix_actions = observed[:, :11], actions[:, :10]
        first = fit_actuator_response(prefix_observed, prefix_actions, preprocessor)
        observed[:, 11:] += 1000.
        actions[:, 10:] -= 2000.
        second = fit_actuator_response(prefix_observed, prefix_actions, preprocessor)
        self.assertTrue(torch.equal(first["M"], second["M"]))
        self.assertTrue(torch.equal(first["singular_values"], second["singular_values"]))
        self.assertEqual(second["num_transitions"], 10)


if __name__ == "__main__":
    unittest.main()
