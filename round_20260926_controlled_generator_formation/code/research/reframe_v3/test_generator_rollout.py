"""Shared observation callbacks extend, rather than bypass, the native cache."""

import unittest
import torch

from research.reframe_v3.test_actuator_rollout import example, legacy_rollout


def test_default_retains_exact_legacy_values_and_action_gradients():
    model, observed, actions = example(3)
    actions.requires_grad_()
    _, expected = legacy_rollout(model, observed, actions)
    _, actual = model.rollout(observed, actions)
    assert torch.equal(actual, expected)
    assert torch.equal(torch.autograd.grad(actual.square().sum(), actions, retain_graph=True)[0],
                       torch.autograd.grad(expected.square().sum(), actions)[0])


def test_shared_callback_sees_native_history_and_correct_chronology():
    model, observed, actions = example(3)
    calls = []
    def transition(history, command):
        calls.append((history.detach().clone(), command.detach().clone()))
        return history[:, -1:, ..., :394] + command.sum(-1, keepdim=True).unsqueeze(2)
    output, packed = model.rollout(observed, actions, observation_transition=transition)
    assert len(calls) == 3
    assert torch.equal(packed[:, :3], model.encode(observed, actions[:, :3]))
    for index, (history, command) in enumerate(calls):
        assert torch.equal(command, actions[:, 2+index:3+index])
        assert torch.equal(history, packed[:, index:index+3])
    assert torch.equal(output['visual'], packed[..., :384])


def test_callback_gradients_and_exclusive_authority():
    model, observed, actions = example()
    def transition(history, command):
        return history[:, -1:, ..., :394] + command.square().sum(-1, keepdim=True).unsqueeze(2)
    actions.requires_grad_()
    loss = model.rollout(observed, actions, observation_transition=transition)[0]['visual'][:, -1].sum()
    derivative = torch.autograd.grad(loss, actions)[0]
    plus, minus = actions.detach().clone(), actions.detach().clone()
    plus[0, 0, 0] += 1e-5
    minus[0, 0, 0] -= 1e-5
    a = model.rollout(observed, plus, observation_transition=transition)[0]['visual'][:, -1].sum()
    b = model.rollout(observed, minus, observation_transition=transition)[0]['visual'][:, -1].sum()
    assert torch.allclose(derivative[0, 0, 0], (a-b)/2e-5, atol=1e-5, rtol=1e-7)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'exclusive'):
        model.rollout(observed, actions, observation_transition=transition,
                      proprio_transition=lambda p, u: p)
    with unittest.TestCase().assertRaisesRegex(ValueError, 'invalid shared'):
        model.rollout(observed, actions, observation_transition=lambda h, u: h)


def load_tests(loader, tests, pattern):
    return unittest.TestSuite(unittest.FunctionTestCase(test) for test in (
        test_default_retains_exact_legacy_values_and_action_gradients,
        test_shared_callback_sees_native_history_and_correct_chronology,
        test_callback_gradients_and_exclusive_authority))


if __name__ == '__main__':
    unittest.main()
