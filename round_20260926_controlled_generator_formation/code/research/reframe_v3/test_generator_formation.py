"""CPU-only chronology and mutation gates for generator formation data."""

import copy
import unittest
from unittest.mock import patch

import torch
from torch import nn

from research.reframe_v3 import generator_formation as formation


class NativeJoinToy:
    num_hist = 3
    concat_dim = 1
    num_proprio_repeat = 1
    num_action_repeat = 1

    def encode_act(self, actions):
        # Nonidentity encoding makes wrong past/current action slots visible.
        return 1000. + 2. * actions


class ScalarStudent(nn.Module):
    kind = "flow"

    def __init__(self):
        super().__init__()
        self.offset = nn.Parameter(torch.tensor(0.))

    def forward(self, history, actions):
        return history[:, -1:, ..., :formation.OBS_DIM] + self.offset


def _numbered(length):
    time = torch.arange(length + 1, dtype=torch.float32)
    observed = {
        "visual": time[None, :, None, None].expand(1, length + 1, 1, 384).clone(),
        "proprio": (100. + time)[None, :, None].expand(1, length + 1, 10).clone(),
    }
    low_time = torch.arange(length, dtype=torch.float32)
    actions = torch.stack((10. * low_time, 10. * low_time + 1.), -1)[None]
    return observed, actions


def _snapshot(dataset):
    return {length: tuple(value.clone() for value in group)
            for length, group in dataset.items()}


class GeneratorFormationTests(unittest.TestCase):
    def assert_dataset_equal(self, actual, expected):
        self.assertEqual(set(actual), set(expected))
        for length in actual:
            for left, right in zip(actual[length], expected[length]):
                self.assertTrue(torch.equal(left, right))

    def test_six_frame_prefix_has_only_zero_to_five_supervision(self):
        observed, low = _numbered(5)
        dataset, times = formation.factual_queries(NativeJoinToy(), observed, low)
        self.assertEqual(times, [{"phase": 0, "start": 0, "end": 5}])
        self.assertEqual(set(dataset), {1})
        history, action, target = dataset[1]
        self.assertEqual(history.shape, (1, 1, 1, 404))
        self.assertEqual(target.shape, (1, 1, 1, 394))
        self.assertTrue(torch.equal(history[..., :384], observed["visual"][:, :1]))
        self.assertTrue(torch.equal(history[:, :, 0, 384:394], observed["proprio"][:, :1]))
        expected_action = low.reshape(1, 1, 10)
        self.assertTrue(torch.equal(action, expected_action))
        self.assertTrue(torch.equal(history[:, :, 0, 394:], 1000. + 2. * expected_action))
        self.assertTrue(torch.equal(target, formation.observation_vector(
            {key: value[:, 5:6] for key, value in observed.items()})))

    def test_full_eleven_frames_have_six_windows_and_ordered_native_history(self):
        observed, low = _numbered(10)
        dataset, times = formation.factual_queries(NativeJoinToy(), observed, low)
        self.assertEqual(times, [
            {"phase": 0, "start": 0, "end": 5},
            {"phase": 0, "start": 5, "end": 10},
            {"phase": 1, "start": 1, "end": 6},
            {"phase": 2, "start": 2, "end": 7},
            {"phase": 3, "start": 3, "end": 8},
            {"phase": 4, "start": 4, "end": 9},
        ])
        self.assertEqual(set(dataset), {1, 2})
        self.assertEqual(sum(group[0].shape[0] for group in dataset.values()), 6)

        history, action, target = dataset[2]
        self.assertEqual(history.shape, (1, 2, 1, 404))
        self.assertTrue(torch.equal(history[:, :, 0, 0], torch.tensor([[0., 5.]])))
        self.assertTrue(torch.equal(history[:, :, 0, 384], torch.tensor([[100., 105.]])))
        two_actions = low.reshape(1, 2, 10)
        self.assertTrue(torch.equal(history[:, :, 0, 394:], 1000. + 2. * two_actions))
        self.assertTrue(torch.equal(action, two_actions[:, 1:2]))
        self.assertTrue(torch.equal(target, formation.observation_vector(
            {key: value[:, 10:11] for key, value in observed.items()})))

        history, action, target = dataset[1]
        self.assertEqual(history.shape, (5, 1, 1, 404))
        for row, start in enumerate(range(5)):
            expected = low[:, start:start + 5].reshape(1, 1, 10)
            self.assertTrue(torch.equal(action[row:row + 1], expected))
            self.assertTrue(torch.equal(history[row:row + 1, :, 0, 394:],
                                        1000. + 2. * expected))
            self.assertEqual(history[row, 0, 0, 0].item(), start)
            self.assertEqual(target[row, 0, 0, 0].item(), start + 5)
            self.assertEqual(target[row, 0, 0, 384].item(), 100. + start + 5)
        # T=1 cross-cut windows retain their own phase, not phase-zero context.
        self.assertTrue(torch.equal(history[:, 0, 0, 0], torch.arange(5).float()))
        self.assertTrue(torch.equal(target[:, 0, 0, 0], torch.arange(5, 10).float()))

    def test_changed_held_tail_cannot_change_explicit_prefix_queries(self):
        observed, low = _numbered(10)
        prefix = {key: value[:, :6].clone() for key, value in observed.items()}
        expected, _ = formation.factual_queries(NativeJoinToy(), prefix, low[:, :5].clone())
        changed = {key: value.clone() for key, value in observed.items()}
        for value in changed.values():
            value[:, 6:] += 10000.
        changed_actions = low.clone()
        changed_actions[:, 5:] -= 5000.
        actual, times = formation.factual_queries(
            NativeJoinToy(), {key: value[:, :6] for key, value in changed.items()},
            changed_actions[:, :5])
        self.assert_dataset_equal(actual, expected)
        self.assertEqual(times, [{"phase": 0, "start": 0, "end": 5}])

    def test_adaptation_changes_only_student_and_preserves_v0_and_input_data(self):
        history = torch.zeros(1, 1, 1, 404)
        action = torch.zeros(1, 1, 10)
        factual = {1: (history.clone(), action.clone(), torch.ones(1, 1, 1, 394))}
        teacher = {1: (history.clone(), action.clone(), torch.zeros(1, 1, 1, 394))}
        held = {1: (history.clone() + 3., action.clone() + 4.,
                    torch.full((1, 1, 1, 394), 8.))}
        snapshots = [_snapshot(dataset) for dataset in (factual, teacher, held)]
        v0 = ScalarStudent().eval()
        vd = copy.deepcopy(v0)
        v0_state = {key: value.clone() for key, value in v0.state_dict().items()}
        sampled = []
        real_sample = formation.sample_batch

        def observed_sample(dataset, random, batch=64):
            sampled.append(id(dataset))
            return real_sample(dataset, random, batch)

        with patch.object(formation, "sample_batch", side_effect=observed_sample):
            rows = formation.adapt_completed(vd, factual, teacher, seed=91, steps=4)
        self.assertEqual(len(rows), 4)
        self.assertEqual(sampled, [id(factual), id(teacher)] * 4)
        self.assertNotIn(id(held), sampled)
        self.assertGreater(vd.offset.item(), 0.)
        self.assertLess(rows[-1]["factual_loss"], rows[0]["factual_loss"])
        self.assertEqual(rows[0]["teacher_loss"], 0.)
        self.assertFalse(vd.training)
        self.assertFalse(v0.training)
        self.assertIsNone(v0.offset.grad)
        for key, value in v0.state_dict().items():
            self.assertTrue(torch.equal(value, v0_state[key]))
        for dataset, snapshot in zip((factual, teacher, held), snapshots):
            self.assert_dataset_equal(dataset, snapshot)


if __name__ == "__main__":
    unittest.main()
