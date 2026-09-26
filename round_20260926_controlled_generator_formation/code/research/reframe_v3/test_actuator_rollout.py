"""CPU structural checks for the optional callback in the real rollout body."""

import unittest
import torch
import torch.nn.functional as F

from models.visual_world_model import VWorldModel


class NativeRolloutToy(torch.nn.Module):
    rollout = VWorldModel.rollout
    concat_dim, num_hist, proprio_dim, action_dim, num_proprio_repeat = 1, 3, 10, 10, 1

    def encode_proprio(self, proprio):
        return F.pad(proprio, (0, 6))

    def encode(self, obs, actions):
        return torch.cat((obs["visual"], self.encode_proprio(obs["proprio"]).unsqueeze(2),
                          actions.unsqueeze(2)), dim=-1)

    def predict(self, source):
        visual = source[..., :384] + .1*source[..., 384:385] + .01*source[..., 394:].sum(-1, keepdim=True)
        return torch.cat((visual, source[..., 384:394]+.3, source[..., 394:]+7.), -1)

    def replace_actions_from_z(self, source, actions):
        return torch.cat((source[..., :394], actions.unsqueeze(2)), -1)

    def separate_emb(self, source):
        return {"visual": source[..., :384], "proprio": source[:, :, 0, 384:394]}, source[..., 394:]


def legacy_rollout(model, observed, actions):
    # Frozen regression reference of the original body, not a deployment path.
    count = observed["visual"].shape[1]
    z = model.encode(observed, actions[:, :count])
    for index in range(count, actions.shape[1]):
        predicted = model.predict(z[:, -model.num_hist:])[:, -1:]
        predicted = model.replace_actions_from_z(predicted, actions[:, index:index+1])
        z = torch.cat((z, predicted), 1)
    predicted = model.predict(z[:, -model.num_hist:])[:, -1:]
    z = torch.cat((z, predicted), 1)
    return model.separate_emb(z)[0], z


def example(initial=1):
    generator = torch.Generator().manual_seed(19)
    observed = {"visual": torch.randn(2, initial, 1, 384, generator=generator, dtype=torch.float64),
                "proprio": torch.randn(2, initial, 4, generator=generator, dtype=torch.float64)}
    actions = torch.randn(2, 5, 10, generator=generator, dtype=torch.float64)
    return NativeRolloutToy(), observed, actions


def advance(proprio, actions):
    summed = actions.reshape(-1, 1, 5, 2).sum(2)
    return proprio + torch.cat((summed, -.5*summed), -1)


class ActuatorRolloutTests(unittest.TestCase):
    def test_joint_visual_revision_readout_feedback_parity(self):
        model, observed, actions = example()
        def revise(visual, predicted, actual):
            return visual + (actual-predicted).mean(-1, keepdim=True)
        base, _ = model.rollout(observed, actions, proprio_transition=advance)
        readout, packed = model.rollout(observed, actions, proprio_transition=advance,
                                       visual_revision=revise, visual_feedback=False)
        feedback, _ = model.rollout(observed, actions, proprio_transition=advance, visual_revision=revise)
        self.assertTrue(torch.equal(readout['proprio'], feedback['proprio']))
        self.assertTrue(torch.equal(readout['proprio'], base['proprio']))
        self.assertTrue(torch.equal(readout['visual'][:, :2], feedback['visual'][:, :2]))
        self.assertFalse(torch.equal(readout['visual'][:, 1], base['visual'][:, 1]))
        self.assertFalse(torch.equal(readout['visual'][:, 2:], feedback['visual'][:, 2:]))
        self.assertTrue(torch.equal(packed[..., :384], readout['visual']))

    def test_joint_visual_revision_gradient_and_invalid_interface(self):
        model, observed, actions = example()
        def revise(visual, predicted, actual):
            return visual + .3*(actual-predicted).sum(-1, keepdim=True)
        with self.assertRaises(ValueError):
            model.rollout(observed, actions, visual_revision=revise)
        with self.assertRaises(ValueError):
            model.rollout(observed, actions, proprio_transition=advance,
                          proprio_feedback=False, visual_revision=revise)
        for feedback in (False, True):
            actions = actions.detach().clone().requires_grad_()
            kwargs = dict(proprio_transition=advance, visual_revision=revise, visual_feedback=feedback)
            prediction = model.rollout(observed, actions, **kwargs)[0]['visual'][:, -1].sum()
            gradient = torch.autograd.grad(prediction, actions)[0]
            plus, minus = actions.detach().clone(), actions.detach().clone()
            plus[0, 0, 0] += 1e-5
            minus[0, 0, 0] -= 1e-5
            finite = (model.rollout(observed, plus, **kwargs)[0]['visual'][:, -1].sum()
                      -model.rollout(observed, minus, **kwargs)[0]['visual'][:, -1].sum())/2e-5
            self.assertAlmostEqual(float(gradient[0, 0, 0]), float(finite), places=6)

    def test_default_exact_legacy_outputs_and_gradients(self):
        for count in (1, 3):
            model, observed, actions = example(count)
            actions.requires_grad_()
            old, old_packed = legacy_rollout(model, observed, actions)
            new, new_packed = model.rollout(observed, actions)
            self.assertTrue(torch.equal(old_packed, new_packed))
            self.assertTrue(all(torch.equal(old[k], new[k]) for k in old))
            old_grad = torch.autograd.grad(old_packed.square().sum(), actions, retain_graph=True)[0]
            new_grad = torch.autograd.grad(new_packed.square().sum(), actions)[0]
            self.assertTrue(torch.equal(old_grad, new_grad))

    def test_same_self_motion_readout_does_not_feed_back(self):
        model, observed, actions = example()
        base, _ = model.rollout(observed, actions)
        readout, readout_packed = model.rollout(observed, actions, proprio_transition=advance, proprio_feedback=False)
        feedback, _ = model.rollout(observed, actions, proprio_transition=advance)
        self.assertTrue(torch.equal(readout["visual"], base["visual"]))
        self.assertTrue(torch.equal(readout["proprio"], feedback["proprio"]))
        self.assertTrue(torch.equal(feedback["visual"][:, :2], base["visual"][:, :2]))
        self.assertFalse(torch.equal(feedback["visual"][:, 2:], base["visual"][:, 2:]))
        self.assertTrue(torch.equal(readout_packed[:, :, 0, 384:394], readout["proprio"]))

    def test_actual_history_skips_old_actions_and_preserves_initial_frames(self):
        model, observed, actions = example(3)
        calls = []
        def record(proprio, action):
            calls.append(action.detach().clone())
            return advance(proprio, action)
        output, _ = model.rollout(observed, actions, proprio_transition=record)
        self.assertEqual(len(calls), 3)
        state = observed["proprio"][:, -1:]
        for offset, action in enumerate(calls):
            self.assertTrue(torch.equal(action, actions[:, 2+offset:3+offset]))
            state = advance(state, action)
            self.assertTrue(torch.equal(output["proprio"][:, 3+offset:4+offset], model.encode_proprio(state)))
        self.assertTrue(torch.equal(output["proprio"][:, :3], model.encode_proprio(observed["proprio"])))

    def test_feedback_action_gradient_matches_finite_difference(self):
        model, observed, actions = example()
        actions.requires_grad_()
        output, _ = model.rollout(observed, actions, proprio_transition=advance)
        gradient = torch.autograd.grad(output["visual"][:, -1].sum(), actions)[0]
        epsilon = 1e-5
        plus, minus = actions.detach().clone(), actions.detach().clone()
        plus[0, 0, 0] += epsilon
        minus[0, 0, 0] -= epsilon
        upper = model.rollout(observed, plus, proprio_transition=advance)[0]["visual"][:, -1].sum()
        lower = model.rollout(observed, minus, proprio_transition=advance)[0]["visual"][:, -1].sum()
        self.assertAlmostEqual(float(gradient[0, 0, 0]), float((upper-lower)/(2*epsilon)), places=6)


if __name__ == "__main__":
    unittest.main()
