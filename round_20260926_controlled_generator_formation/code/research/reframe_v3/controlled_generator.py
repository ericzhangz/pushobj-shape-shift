"""Bounded controlled-flow and ordered packed-network formation references.

This is a standard NCDE-style numerical reference, not an originality claim or
an inferred physical microstate.  A native history supplies the latest encoded
observation and a fixed context of past complete tokens.  The flow consumes
each of the five two-dimensional commands in order through shared drift and
control fields; it never puts the current chunk in its context.  Native cache
composition, teacher fitting and information-boundary enforcement belong to
the caller.

``configure_normalization`` fixes coordinates from legal teacher inputs, not
from forward calls.  Its 1e-3 scale floor prevents division by a zero/near-zero
teacher standard deviation.  It changes conditioning, not the physical law;
the floor is explicitly stress-tested and all outputs retain native units.
"""

import torch
from torch import nn


class ControlledGenerator(nn.Module):
    RAW_CHUNK_DIM = 10
    COMMANDS_PER_CHUNK = 5
    MIN_OBSERVATION_SCALE = 1e-3

    def __init__(self, obs_dim=394, action_emb_dim=10, num_hist=3,
                 hidden=128, kind="flow"):
        super().__init__()
        for name, value in (("obs_dim", obs_dim), ("action_emb_dim", action_emb_dim),
                            ("num_hist", num_hist), ("hidden", hidden)):
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ValueError(f"{name} must be a positive integer")
        if kind not in ("flow", "packed"):
            raise ValueError("kind must be 'flow' or 'packed'")
        self.obs_dim = obs_dim
        self.action_emb_dim = action_emb_dim
        self.num_hist = num_hist
        self.hidden = hidden
        self.kind = kind
        self.context_dim = (num_hist - 1) * (obs_dim + action_emb_dim + 1)
        self.register_buffer("observation_center", torch.zeros(obs_dim))
        self.register_buffer("observation_scale", torch.ones(obs_dim))

        input_dim = self.context_dim + obs_dim
        if kind == "flow":
            self.fields = self._mlp(input_dim, 3 * obs_dim)
        else:
            self.packed_net = self._mlp(input_dim + self.RAW_CHUNK_DIM, obs_dim)

    def _mlp(self, input_dim, output_dim):
        network = nn.Sequential(
            nn.Linear(input_dim, self.hidden), nn.Tanh(),
            nn.Linear(self.hidden, self.hidden), nn.Tanh(),
            nn.Linear(self.hidden, output_dim),
        )
        # Both references start at identity, not at the frozen predictor F0.
        nn.init.zeros_(network[-1].weight)
        nn.init.zeros_(network[-1].bias)
        return network

    def configure_normalization(self, center, scale):
        """Copy fixed [obs_dim] teacher statistics into persistent buffers.

        Nonnegative scales are clamped at 1e-3.  Negative/nonfinite statistics
        fail rather than concealing a broken teacher or using a fallback.
        The caller must compute statistics only from its permitted evidence.
        """
        for name, value in (("center", center), ("scale", scale)):
            if (not isinstance(value, torch.Tensor) or value.shape != (self.obs_dim,)
                    or not value.is_floating_point() or not torch.isfinite(value).all()):
                raise ValueError(f"{name} must be a finite floating [obs_dim] tensor")
        if (scale < 0).any():
            raise ValueError("observation scale cannot be negative")
        with torch.no_grad():
            self.observation_center.copy_(center.detach().to(self.observation_center))
            self.observation_scale.copy_(scale.detach().to(self.observation_scale)
                                         .clamp_min(self.MIN_OBSERVATION_SCALE))
        return self

    def parameter_count(self):
        """Actual parameter count; the two arms are not exactly matched."""
        return sum(parameter.numel() for parameter in self.parameters())

    def _validate_history(self, history_z):
        width = self.obs_dim + self.action_emb_dim
        if (not isinstance(history_z, torch.Tensor) or history_z.ndim != 4
                or history_z.shape[0] < 1 or history_z.shape[1] < 1
                or history_z.shape[2:] != (1, width)):
            raise ValueError("history_z must have shape [B,T,1,obs_dim+action_emb_dim]")
        self._validate_tensor(history_z, "history_z")

    def _validate_tensor(self, value, name):
        if not value.is_floating_point() or not torch.isfinite(value).all():
            raise ValueError(f"{name} must be finite and floating point")
        if (value.device != self.observation_center.device
                or value.dtype != self.observation_center.dtype):
            raise ValueError(f"{name} must match the module buffer device and dtype")

    def build_context(self, history_z):
        """Past complete z tokens plus per-frame valid masks, left padded.

        All slots of the latest token's action embedding are excluded.  Older
        native tokens retain their past action embeddings, and observations
        use the fixed scaling.  Extra-old frames are truncated as in the
        native history cache.  Zero padding is applied after normalization.
        """
        self._validate_history(history_z)
        batch, length = history_z.shape[:2]
        past_count = self.num_hist - 1
        if past_count == 0:
            return history_z.new_zeros((batch, 0))
        past = history_z[:, max(0, length - self.num_hist):length - 1, 0]
        observations = ((past[..., :self.obs_dim] - self.observation_center)
                        / self.observation_scale)
        past = torch.cat((observations, past[..., self.obs_dim:],
                          past.new_ones((*past.shape[:2], 1))), dim=-1)
        padding = past.new_zeros((batch, past_count - past.shape[1],
                                  self.obs_dim + self.action_emb_dim + 1))
        return torch.cat((padding, past), dim=1).reshape(batch, self.context_dim)

    def vector_fields(self, context, observation):
        """Return shared [B,3,obs_dim] drift/control fields in native units."""
        if self.kind != "flow":
            raise ValueError("vector_fields is only available for kind='flow'")
        if context.shape != (observation.shape[0], self.context_dim):
            raise ValueError("invalid context shape")
        if observation.shape != (context.shape[0], self.obs_dim):
            raise ValueError("invalid observation shape")
        self._validate_tensor(context, "context")
        self._validate_tensor(observation, "observation")
        coordinates = (observation - self.observation_center) / self.observation_scale
        fields = self.fields(torch.cat((context, coordinates), dim=-1))
        if fields.shape != (context.shape[0], 3 * self.obs_dim):
            raise ValueError("field network returned an invalid shape")
        fields = fields.reshape(context.shape[0], 3, self.obs_dim) * self.observation_scale
        if not torch.isfinite(fields).all():
            raise FloatingPointError("nonfinite controlled field")
        return fields

    def _velocity(self, context, observation, command):
        fields = self.vector_fields(context, observation)
        return (fields[:, 0] + command[:, :1] * fields[:, 1]
                + command[:, 1:] * fields[:, 2])

    def forward(self, history_z, packed_actions, *, substeps=1):
        self._validate_history(history_z)
        if (not isinstance(packed_actions, torch.Tensor)
                or packed_actions.shape != (history_z.shape[0], 1, self.RAW_CHUNK_DIM)):
            raise ValueError("packed_actions must have shape [B,1,10]")
        self._validate_tensor(packed_actions, "packed_actions")
        if isinstance(substeps, bool) or not isinstance(substeps, int) or substeps not in (1, 2):
            raise ValueError("substeps must be 1 or 2")
        if self.kind == "packed" and substeps != 1:
            raise ValueError("the packed reference has no substep integrator")
        context = self.build_context(history_z)
        observation = history_z[:, -1, 0, :self.obs_dim]
        if self.kind == "packed":
            coordinates = ((observation - self.observation_center)
                           / self.observation_scale)
            increment = self.packed_net(torch.cat(
                (context, coordinates, packed_actions[:, 0]), dim=-1))
            if increment.shape != observation.shape:
                raise ValueError("packed network returned an invalid shape")
            observation = observation + increment * self.observation_scale
        else:
            commands = packed_actions[:, 0].reshape(-1, self.COMMANDS_PER_CHUNK, 2)
            dt = 1.0 / (self.COMMANDS_PER_CHUNK * substeps)
            for command in commands.unbind(dim=1):
                for _ in range(substeps):
                    first = self._velocity(context, observation, command)
                    midpoint = observation + (0.5 * dt) * first
                    second = self._velocity(context, midpoint, command)
                    observation = observation + dt * second
        if not torch.isfinite(observation).all():
            raise FloatingPointError("nonfinite controlled observation")
        return observation[:, None, None]
