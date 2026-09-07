"""Representation-learning objectives compared by the Part-3 intervention.

Part 3 varies only the observation representation ``E`` that feeds the shared
transition-novelty metric ``N_T``. The three compared objectives are distinct
*kinds* of representation learning, not interchangeable encoders:

* ``bigan`` -- adversarial generative representation. ``E`` is the BiGAN
  encoder, trained jointly with a generator and a joint discriminator so that
  the encoded ``(x, E(x))`` joint distribution is indistinguishable from the
  generated ``(G(z), z)`` distribution.

* ``idf`` -- inverse-dynamics / action-predictive representation. ``E`` is
  trained so that the action ``a_t`` is predictable from ``(E(s_t), E(s_{t+1}))``.

* ``rnd`` -- random-target prediction representation. ``E`` is the output of a
  predictor network trained to match a frozen, randomly initialised target
  network ``T`` on the observation, i.e. ``E(s) ~= T(s)`` on the training
  distribution.

Each representation exposes an ``encoder`` module (the map ``E``) and a
``train`` method that performs one online update step. The forward-model and
transition-novelty modules consume only ``encoder``, so the intervention is
isolated to ``E``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from bigan.discriminator import BiGANDiscriminator
from bigan.encoder import BiGANEncoder
from bigan.generator import BiGANGenerator
from bigan.trainer import BiGANTrainer, BiGANUpdateMetrics
from config import BiGANConfig, Part3Config
from utils.replay import ReplayBuffer
from utils.transition_replay import TransitionBatch, TransitionReplayBuffer
from part3.rng import (
    make_generator,
    run_under_rng,
    RNG_BIGAN_LATENT,
    RNG_REPRESENTATION_INIT,
    RNG_REPLAY_SAMPLING,
    RNG_RND_TARGET,
)


@dataclass(frozen=True)
class RepresentationUpdateMetrics:
    """Diagnostics from one online representation-training step."""

    updated: bool
    loss: float
    update_count: int


@dataclass(frozen=True)
class RepresentationDescription:
    """Frozen description of one representation for reporting/auditing."""

    name: str
    architecture: str
    latent_dimension: int
    parameter_count: int
    objective: str
    optimizer: str
    learning_rate: float
    batch_size: int
    update_frequency: str
    initialization: str
    training_data: str
    number_of_updates: int
    online: bool


def _param_count(module: nn.Module) -> int:
    """Sum the number of scalar parameters in a module.

    Input: A ``torch.nn.Module``.
    Output: The total parameter count.
    Mathematical meaning: Measures representation capacity for the audit.
    """
    return int(sum(parameter.numel() for parameter in module.parameters()))


def _build_encoder(
    base_seed: int,
    observation_shape: tuple[int, ...],
    config: Part3Config,
    device: Union[torch.device, str],
    stream_id: int = RNG_REPRESENTATION_INIT,
) -> BiGANEncoder:
    """Build one representation encoder under a dedicated init stream.

    Input: Base seed, observation shape, Part-3 configuration, device, and the
        stream id used to seed the orthogonal initialisation.
    Output: An orthogonally-initialised ``BiGANEncoder``.
    Mathematical meaning: All three representations reuse one architecture so
        that only the learning objective differs across arms.
    """
    return run_under_rng(
        base_seed,
        stream_id,
        lambda: BiGANEncoder(
            observation_shape,
            config.latent_dim,
            config.feature_dim,
            config.hidden_dim,
        ),
    ).to(device)


class BiGANRepresentation:
    """Adversarial generative representation (BiGAN encoder)."""

    def __init__(
        self,
        base_seed: int,
        observation_shape: tuple[int, ...],
        config: Part3Config,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Construct the BiGAN representation and its adversarial trainer.

        Input: Base seed, observation shape, Part-3 configuration, and device.
        Output: A representation whose ``encoder`` is the BiGAN encoder and
            whose ``train`` runs one adversarial update.
        Mathematical meaning: ``E`` is optimised by the bidirectional
            adversarial objective, which makes it a generative rather than a
            purely predictive code.
        """
        self.name = "bigan"
        self.base_seed = base_seed
        self.device = torch.device(device)
        self.encoder = _build_encoder(base_seed, observation_shape, config, device)
        self.generator = run_under_rng(
            base_seed,
            RNG_REPRESENTATION_INIT,
            lambda: BiGANGenerator(
                observation_shape,
                config.latent_dim,
                config.feature_dim,
                config.hidden_dim,
            ),
        ).to(device)
        self.discriminator = run_under_rng(
            base_seed,
            RNG_REPRESENTATION_INIT,
            lambda: BiGANDiscriminator(
                observation_shape,
                config.latent_dim,
                config.feature_dim,
                config.hidden_dim,
            ),
        ).to(device)
        bigan_config = BiGANConfig(
            latent_dim=config.latent_dim,
            feature_dim=config.feature_dim,
            hidden_dim=config.hidden_dim,
            learning_rate=config.representation_learning_rate,
            # Use the same Adam betas as the IDF/RND arm so no representation is
            # given a separately tuned optimizer: only the objective differs.
            adam_beta1=0.9,
            adam_beta2=0.999,
            batch_size=config.batch_size,
            update_interval=config.representation_update_interval,
        )
        self.trainer = BiGANTrainer(
            self.encoder,
            self.generator,
            self.discriminator,
            bigan_config,
            observation_shape,
            self.device,
        )
        self.update_count = 0
        self.config = config

    def train(self, replay_buffer: ReplayBuffer) -> RepresentationUpdateMetrics:
        """Run one adversarial representation update under a dedicated stream.

        Input: Observation replay buffer.
        Output: Representation update metrics.
        Mathematical meaning: The legacy BiGAN trainer draws replay minibatches
            and prior latents from the (implicit) global RNG, so the whole
            update is executed inside a forked RNG scope seeded from the
            dedicated BiGAN latent/replay stream.
        """
        if len(replay_buffer) < self.config.batch_size:
            return RepresentationUpdateMetrics(False, 0.0, self.update_count)
        metrics = run_under_rng(
            self.base_seed,
            RNG_BIGAN_LATENT,
            lambda: self.trainer.update(replay_buffer),
        )
        if isinstance(metrics, BiGANUpdateMetrics):
            self.update_count = metrics.update_count
            loss = metrics.discriminator_loss + metrics.encoder_generator_loss
            return RepresentationUpdateMetrics(metrics.updated, float(loss), self.update_count)
        return RepresentationUpdateMetrics(False, 0.0, self.update_count)


class IDFRepresentation:
    """Inverse-dynamics / action-predictive representation.

    The encoder ``E`` is trained so that, given the paired latent codes of two
    consecutive states, a head can recover the action that caused the
    transition. The head is an auxiliary used only to train ``E``; it is not
    part of the transition-novelty metric.
    """

    def __init__(
        self,
        base_seed: int,
        observation_shape: tuple[int, ...],
        action_dim: int,
        discrete_actions: bool,
        config: Part3Config,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Construct the inverse-dynamics head and its encoder.

        Input: Base seed, observation shape, action dimension, action type,
            Part-3 configuration, and device.
        Output: An IDF representation with an ``encoder`` and a ``train`` step.
        Mathematical meaning: ``E`` is optimised by the action-prediction
            objective ``a_t ~ g(E(s_t), E(s_{t+1}))``.
        """
        self.name = "idf"
        self.base_seed = base_seed
        self.device = torch.device(device)
        self.action_dim = action_dim
        self.discrete_actions = discrete_actions
        self.encoder = _build_encoder(base_seed, observation_shape, config, device)
        hidden = config.idf_hidden_dim
        self.head = nn.Sequential(
            nn.Linear(2 * config.latent_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
        )
        # Initialise the head orthogonally under the representation-init stream.
        run_under_rng(
            base_seed,
            RNG_REPRESENTATION_INIT,
            lambda: self.head.apply(self._init_head),
        )
        self.head = self.head.to(device)
        self.optimizer = torch.optim.Adam(
            list(self.encoder.parameters()) + list(self.head.parameters()),
            lr=config.representation_learning_rate,
        )
        # Replay buffers are stored on CPU in the Part-3 trainer, so the replay
        # sampling generator lives on CPU too.
        self.replay_generator = make_generator(base_seed, RNG_REPLAY_SAMPLING, "cpu")
        self.update_count = 0
        self.config = config

    def _init_head(self, module: nn.Module) -> None:
        """Orthogonally initialise the inverse-dynamics head layers.

        Input: A module of the head.
        Output: No value; weights are initialised in place.
        Mathematical meaning: Gives the action head a stable initial Jacobian.
        """
        if isinstance(module, nn.Linear):
            nn.init.orthogonal_(module.weight)
            nn.init.zeros_(module.bias)

    def train(self, transitions: TransitionReplayBuffer) -> RepresentationUpdateMetrics:
        """Run one inverse-dynamics update under a dedicated stream.

        Input: Transition replay buffer.
        Output: Representation update metrics.
        Mathematical meaning: Minimises action-prediction loss, which shapes
            ``E`` so that consecutive latent codes are action-predictive.
        """
        if len(transitions) < self.config.batch_size:
            return RepresentationUpdateMetrics(False, 0.0, self.update_count)
        batch: TransitionBatch = transitions.sample(
            self.config.batch_size, generator=self.replay_generator
        )
        observations = batch.observations.to(self.device).float()
        next_observations = batch.next_observations.to(self.device).float()
        actions = batch.actions.to(self.device)
        # Inverse dynamics: predict a_t from (E(s_t), E(s_{t+1})). Both latent
        # codes receive gradients (the standard ICM formulation), so the
        # encoder is shaped by the action-predictive objective.
        codes = self.encoder(observations)
        next_codes = self.encoder(next_observations)
        logits = self.head(torch.cat((codes, next_codes), dim=-1))
        if self.discrete_actions:
            action_targets = actions.long().reshape(-1)
            loss = F.cross_entropy(logits, action_targets)
        else:
            action_targets = actions.float()
            loss = F.mse_loss(logits, action_targets)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.optimizer.param_groups[0]["params"], 0.5)
        self.optimizer.step()
        self.update_count += 1
        return RepresentationUpdateMetrics(True, float(loss.detach().cpu()), self.update_count)


class RNDRepresentation:
    """Random-target prediction representation.

    A predictor network ``P`` (the representation ``E``) is trained to match the
    output of a frozen, randomly initialised target network ``T``:

    ``E(s) = P(s),  loss = ||P(s) - T(s)||_2^2``

    The target is created under the dedicated RND-target stream and is never
    updated. Only the predictor matters for ``N_T``.
    """

    def __init__(
        self,
        base_seed: int,
        observation_shape: tuple[int, ...],
        config: Part3Config,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Construct the RND predictor, frozen target, and optimizer.

        Input: Base seed, observation shape, Part-3 configuration, and device.
        Output: An RND representation with an ``encoder`` (the predictor) and a
            ``train`` step.
        Mathematical meaning: ``E`` is optimised by a regression to a fixed
            random feature map ``T``, producing a stochastic-feature code.
        """
        self.name = "rnd"
        self.base_seed = base_seed
        self.device = torch.device(device)
        self.encoder = _build_encoder(base_seed, observation_shape, config, device)
        self.target = _build_encoder(
            base_seed, observation_shape, config, device, stream_id=RNG_RND_TARGET
        )
        for parameter in self.target.parameters():
            parameter.requires_grad_(False)
        self.target.eval()
        self.optimizer = torch.optim.Adam(
            self.encoder.parameters(),
            lr=config.representation_learning_rate,
        )
        self.replay_generator = make_generator(base_seed, RNG_REPLAY_SAMPLING, "cpu")
        self.update_count = 0
        self.config = config

    def train(self, replay_buffer: ReplayBuffer) -> RepresentationUpdateMetrics:
        """Run one RND update under a dedicated stream.

        Input: Observation replay buffer.
        Output: Representation update metrics.
        Mathematical meaning: Minimises the mean squared distance to the frozen
            random target on the observed state distribution.
        """
        if len(replay_buffer) < self.config.batch_size:
            return RepresentationUpdateMetrics(False, 0.0, self.update_count)
        batch = replay_buffer.sample(
            self.config.batch_size,
            generator=self.replay_generator,
            device=self.device,
        ).observations
        with torch.no_grad():
            target_values = self.target(batch).detach()
        prediction = self.encoder(batch)
        loss = (prediction - target_values).square().mean()
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.encoder.parameters(), 0.5)
        self.optimizer.step()
        self.update_count += 1
        return RepresentationUpdateMetrics(True, float(loss.detach().cpu()), self.update_count)

    @property
    def target_rng_stream(self) -> int:
        """Return the stream id used to initialise the random target."""
        return RNG_RND_TARGET


def build_representation(
    base_seed: int,
    observation_shape: tuple[int, ...],
    representation: str,
    config: Part3Config,
    action_dim: int,
    discrete_actions: bool,
    device: Union[torch.device, str] = "cpu",
) -> object:
    """Factory for one Part-3 representation.

    Input: Base seed, observation shape, representation name, configuration,
        action dimension, action type, and device.
    Output: A representation object exposing ``encoder`` and ``train``.
    Mathematical meaning: Selects the representation-backbone under test.
    """
    if representation == "bigan":
        return BiGANRepresentation(base_seed, observation_shape, config, device)
    if representation == "idf":
        return IDFRepresentation(
            base_seed, observation_shape, action_dim, discrete_actions, config, device
        )
    if representation == "rnd":
        return RNDRepresentation(base_seed, observation_shape, config, device)
    raise ValueError("representation must be 'bigan', 'idf', or 'rnd'")


def describe_representation(representation: object, config: Part3Config) -> RepresentationDescription:
    """Return an auditable description of a representation.

    Input: A built representation object and its Part-3 configuration.
    Output: A frozen description including architecture, latent dimension,
        parameter count, objective, optimizer, and schedule.
    Mathematical meaning: Documents the intervention arms so the causal test is
        auditable and no arm is tuned differently from another.
    """
    name = getattr(representation, "name")
    encoder = representation.encoder
    latent_dim = int(encoder.latent_dim)
    live_updates = int(getattr(representation, "update_count", 0))
    if name == "bigan":
        architecture = (
            "BiGAN conv/MLP encoder (feature trunk -> "
            f"latent head), latent_dim={latent_dim}"
        )
        objective = "adversarial generative (BiGAN joint distribution matching)"
        optimizer = "Adam (separate encoder/generator/discriminator optimizers)"
    elif name == "idf":
        architecture = (
            "same encoder trunk as BiGAN (feature trunk -> latent head) + "
            "inverse-dynamics action head"
        )
        objective = "inverse-dynamics / action-predictive (predict a_t from E(s_t),E(s_{t+1}))"
        optimizer = "Adam (encoder + head)"
    elif name == "rnd":
        architecture = (
            "same encoder trunk as BiGAN (feature trunk -> latent head) + frozen "
            "random target network"
        )
        objective = "random-target prediction (match frozen target T(s))"
        optimizer = "Adam (predictor only; target frozen)"
    else:
        raise ValueError("unknown representation")
    return RepresentationDescription(
        name=name,
        architecture=architecture,
        latent_dimension=latent_dim,
        parameter_count=_param_count(encoder),
        objective=objective,
        optimizer=optimizer,
        learning_rate=config.representation_learning_rate,
        batch_size=config.batch_size,
        update_frequency=f"every {config.representation_update_interval} PPO update(s)",
        initialization="orthogonal (scoped to RNG_REPRESENTATION_INIT)",
        training_data=(
            "observation replay (BiGAN, RND) / transition replay (IDF), "
            "same rollout stream and capacity"
        ),
        number_of_updates=live_updates,
        online=True,
    )
