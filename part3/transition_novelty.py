"""Pure Part-3 transition novelty and shared forward-model training.

This module implements the *only* transition novelty used by Part 3, explicitly
and exactly as::

    N_T(s_t, a_t, s_{t+1}) = || f(E(s_t), a_t) - E(s_{t+1}) ||_2

where ``E`` is the representation under test (BiGAN, IDF, or RND) and ``f`` is
one shared latent forward model ``LatentForwardModel``.

The metric is the L2 norm (``||.||_2``) over the latent vector. The forward
model is trained by the squared-L2 / MSE regression::

    loss = mean( || f(E(s_t), a_t) - E(s_{t+1}) ||_2^2 )

This loss is identical across all three representations. There is no L1 term
anywhere in the Part-3 transition path.

The Part-3 path deliberately contains none of the following: BiGAN
discriminator features, generator reconstruction, feature matching, EME, kNN,
visit counts, episodic counts, ensemble uncertainty, adaptive scaling, state
novelty, or any death/respawn bonus. Death/respawn transitions are neither
masked nor rewarded here; they are only logged by the trainer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import torch
from torch import Tensor, nn

from exploration.transition_novelty import LatentForwardModel
from utils.normalization import RunningMeanVariance, RunningNormalizer
from utils.transition_replay import TransitionBatch, TransitionReplayBuffer
from part3.rng import make_generator, RNG_FORWARD_TRAIN, RNG_REPLAY_SAMPLING


@dataclass(frozen=True)
class Part3TransitionComponents:
    """Per-transition results of the pure Part-3 novelty function.

    ``novelty`` is the exact ``N_T`` L2 metric. ``normalized_score`` is the
    shared Eq. (5) reward processing applied identically across the three
    representations. ``combined_score`` and the ``pixel_error``/``feature_error``
    aliases are provided so this object satisfies the generic novelty contract
    used by the trainer, reward pipeline, and logger.
    """

    novelty: Tensor
    normalized_score: Tensor

    @property
    def combined_score(self) -> Tensor:
        """Return the raw ``N_T`` under the generic novelty name."""
        return self.novelty

    @property
    def pixel_error(self) -> Tensor:
        """Return ``N_T`` under the generic ``pixel_error`` alias."""
        return self.novelty

    @property
    def feature_error(self) -> Tensor:
        """Return zeros under the generic ``feature_error`` alias.

        Part 3 performs no discriminator feature matching, so this term is
        identically zero and never contributes to the novelty or the reward.
        """
        return self.novelty.new_zeros(self.novelty.shape)


class Part3TransitionNoveltyEstimator:
    """Compute ``N_T = ||f(E(s),a) - E(s')||_2`` and Eq. (5) reward processing.

    Args:
        encoder: The representation map ``E`` (BiGAN, IDF, or RND encoder).
        forward_model: The shared latent forward model ``f``.
        normalization_epsilon: Stability constant for the running variance.
        normalization_clip: Optional symmetric clip on the normalized score.
        device: Computation device.
    """

    def __init__(
        self,
        encoder: nn.Module,
        forward_model: LatentForwardModel,
        normalization_epsilon: float = 1.0e-8,
        normalization_clip: Optional[float] = 5.0,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Store the representation and forward model and create Eq. (5) stats.

        Input: Representation encoder, shared forward model, normalization
            settings, and computation device.
        Output: A pure Part-3 transition novelty estimator.
        Mathematical meaning: Defines ``N_T`` over the latent metric and the
            online reward-scale normalization shared by all three arms.
        """
        self.encoder = encoder.to(device)
        self.forward_model = forward_model.to(device)
        self.device = torch.device(device)
        self.normalizer = RunningNormalizer((), normalization_epsilon, normalization_clip, self.device)
        self.extrinsic_statistics = RunningMeanVariance((), normalization_epsilon, self.device)

    @torch.no_grad()
    def score(
        self,
        observations: Tensor,
        actions: Tensor,
        next_observations: Tensor,
    ) -> Part3TransitionComponents:
        """Compute the exact pure L2 transition novelty.

        Input: ``s_t``, ``a_t``, and ``s_{t+1}`` batches.
        Output: Raw ``N_T`` and Eq. (5)-processed normalized score.
        Mathematical meaning: Encodes both endpoints, predicts the next code
            with ``f``, takes the L2 norm of the residual, then applies the
            shared reward-scale normalization.
        """
        observations = observations.to(self.device).float()
        next_observations = next_observations.to(self.device).float()
        actions = actions.to(self.device)
        latent = self.encoder(observations)
        next_latent = self.encoder(next_observations)
        predicted = self.forward_model(latent, actions)
        residual = predicted - next_latent
        N_T = residual.norm(dim=1, p=2)
        if self.extrinsic_statistics.count == 0.0:
            extrinsic_mean = torch.zeros((), device=self.device)
        else:
            extrinsic_mean = self.extrinsic_statistics.mean
        normalized = self.normalizer.normalize(
            N_T,
            update=True,
            mean_offset=extrinsic_mean,
        )
        return Part3TransitionComponents(novelty=N_T, normalized_score=normalized)

    def __call__(
        self,
        observations: Tensor,
        actions: Tensor,
        next_observations: Tensor,
        extrinsic_reward: Tensor,
    ) -> Part3TransitionComponents:
        """Compute Part-3 transition novelty with the current reward scale.

        Input: Transition batch and matching extrinsic rewards.
        Output: Part-3 transition novelty components.
        Mathematical meaning: Updates the running extrinsic-reward mean used by
            Eq. (5) and then evaluates ``N_T``.
        """
        self.extrinsic_statistics.update(extrinsic_reward.reshape(-1))
        return self.score(observations, actions, next_observations)


@dataclass(frozen=True)
class ForwardModelUpdateMetrics:
    """Diagnostics from one Part-3 forward-model update."""

    updated: bool
    mse_loss: float
    update_count: int


class Part3ForwardModelTrainer:
    """Fit the shared forward model with the squared-L2 / MSE objective.

    The objective is exactly ``mean(||f(E(s_t),a_t) - E(s_{t+1})||_2^2)`` for
    every representation. Only ``f`` is optimized; the encoder ``E`` is used as
    a frozen target and never receives gradients here.
    """

    def __init__(
        self,
        estimator: Part3TransitionNoveltyEstimator,
        learning_rate: float,
        update_epochs: int,
        max_grad_norm: float,
    ) -> None:
        """Initialize the forward-model optimizer under a dedicated stream.

        Input: Part-3 estimator, forward-model learning rate, update epochs, and
            gradient clip.
        Output: A trainer that updates only ``f``.
        Mathematical meaning: Defines the regression process that fits
            ``f(E(s_t), a_t) ~= E(s_{t+1})`` with an MSE objective.
        """
        self.estimator = estimator
        self.optimizer = torch.optim.Adam(
            estimator.forward_model.parameters(), lr=learning_rate
        )
        self.update_epochs = update_epochs
        self.max_grad_norm = max_grad_norm
        self.replay_generator = make_generator(
            # Base seed is seeded by the trainer; stream id fixed here.
            0,
            RNG_REPLAY_SAMPLING,
            "cpu",
        )
        self.update_count = 0

    def set_base_seed(self, base_seed: int) -> None:
        """Bind the training stream generator to the experiment seed.

        Input: Experiment base seed.
        Output: No value; regenerates the replay-sampling generator.
        Mathematical meaning: Preserves stream isolation while following the
            experiment seed.
        """
        self.replay_generator = make_generator(base_seed, RNG_REPLAY_SAMPLING, "cpu")

    def update(self, replay_buffer: TransitionReplayBuffer, batch_size: int) -> ForwardModelUpdateMetrics:
        """Train the forward model on replayed latent transitions (MSE).

        Input: Transition replay buffer and minibatch size.
        Output: Forward-model diagnostics.
        Mathematical meaning: Minimizes
            ``mean(||f(E(s),a)-E(s')||_2^2)``; only ``f`` is updated.
        """
        if len(replay_buffer) < batch_size:
            return ForwardModelUpdateMetrics(False, 0.0, self.update_count)
        losses: list[Tensor] = []
        for _ in range(self.update_epochs):
            batch: TransitionBatch = replay_buffer.sample(
                batch_size, generator=self.replay_generator
            )
            states = batch.observations.to(self.estimator.device).float()
            next_states = batch.next_observations.to(self.estimator.device).float()
            actions = batch.actions.to(self.estimator.device)
            with torch.no_grad():
                source = self.estimator.encoder(states)
                target = self.estimator.encoder(next_states)
            predicted = self.estimator.forward_model(source, actions)
            residual = predicted - target
            # Squared L2 = sum over latent dim of (pred - target)^2; mean over batch.
            loss = residual.square().sum(dim=1).mean()
            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                self.estimator.forward_model.parameters(), self.max_grad_norm
            )
            self.optimizer.step()
            losses.append(loss.detach())
        self.update_count += 1
        return ForwardModelUpdateMetrics(
            True,
            float(torch.stack(losses).mean().cpu()),
            self.update_count,
        )
