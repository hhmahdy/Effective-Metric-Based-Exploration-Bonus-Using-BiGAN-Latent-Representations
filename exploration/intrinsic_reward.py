"""Intrinsic-reward conversion without an extra scaling coefficient.

Equation (5) already places normalized novelty on the extrinsic-reward scale.
The single beta coefficient is applied later to the intrinsic advantage inside
PPO, never here.

Two novelty sources can feed this module:

* Adventurer's reconstruction novelty ``B(s)`` from :mod:`exploration.novelty`
  or its transition variant, converted by :class:`IntrinsicRewardPipeline`.
* The metric-based exploration bonus of Contribution 2,
  ``b_t = ||E(s_t)-E(s_{t+1})||_p * min(max(zeta(r), 1), M)``, produced by
  :class:`MetricIntrinsicReward`.

Both paths emit the same structural contract -- a components object exposing
``combined_score`` and ``normalized_score`` -- so the trainer, episodic memory,
and logger treat them interchangeably.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Union

import torch
from torch import Tensor

from exploration.ensemble_scaling import EnsembleRewardVariance
from exploration.novelty import NormalizedNoveltyScore, NoveltyComponents
from exploration.state_discrepancy import LatentStateDiscrepancy


NoveltyLike = Any


@dataclass(frozen=True)
class IntrinsicRewardResult:
    """Separate extrinsic and intrinsic reward streams."""

    extrinsic_reward: Tensor
    intrinsic_reward: Tensor
    normalized_novelty: Tensor


class IntrinsicRewardCalculator:
    """Return Eq. (5) novelty as intrinsic reward without rescaling."""

    def __init__(self, clip_value: Optional[float] = None) -> None:
        """Initialize optional safety clipping only.

        Input: Optional positive clipping bound. No multiplicative coefficient
            is accepted because Eq. (5) already performs reward scaling.
        Output: Intrinsic reward converter.
        Mathematical meaning: Implements ``r_i = normalized_novelty``.
        """
        if clip_value is not None and clip_value <= 0.0:
            raise ValueError("clip_value must be positive when provided")
        self.clip_value = clip_value

    def __call__(self, normalized_novelty: Tensor) -> Tensor:
        """Return normalized novelty as intrinsic reward.

        Input: Eq. (5)-normalized novelty tensor.
        Output: Intrinsic reward tensor with identical values unless optional
            safety clipping is configured.
        Mathematical meaning: Computes ``r_t^i=r_t^i(s_t)`` with no beta.
        """
        intrinsic_reward = normalized_novelty
        if self.clip_value is not None:
            intrinsic_reward = intrinsic_reward.clamp(-self.clip_value, self.clip_value)
        return intrinsic_reward


class IntrinsicRewardPipeline:
    """Produce separate extrinsic and Eq. (5) intrinsic rewards."""

    def __init__(self, clip_value: Optional[float] = None) -> None:
        """Construct the reward converter.

        Input: Optional safety clip only.
        Output: Pipeline that never mixes or multiplicatively scales rewards.
        Mathematical meaning: Leaves beta exclusively in advantage combination.
        """
        self.calculator = IntrinsicRewardCalculator(clip_value)

    def __call__(self, extrinsic_reward: Tensor, novelty: "NoveltyLike") -> IntrinsicRewardResult:
        """Return independent reward streams.

        Input: Environment reward and any novelty result exposing
            ``normalized_score`` -- reconstruction novelty, transition novelty,
            or the metric bonus -- with matching batch shape.
        Output: Separate extrinsic reward, intrinsic reward, and normalized score.
        Mathematical meaning: Produces ``r_t^e`` unchanged and ``r_t^i`` from
            Eq. (5), with no pre-GAE mixture.
        """
        normalized_novelty = novelty.normalized_score
        if extrinsic_reward.shape != normalized_novelty.shape:
            raise ValueError("extrinsic reward and novelty shapes must match")
        return IntrinsicRewardResult(
            extrinsic_reward=extrinsic_reward,
            intrinsic_reward=self.calculator(normalized_novelty),
            normalized_novelty=normalized_novelty,
        )


@dataclass(frozen=True)
class MetricNoveltyComponents:
    """Per-transition components of the metric-based exploration bonus.

    The ``pixel_error`` and ``feature_error`` properties are compatibility
    aliases so this object can be logged and stored by the same trainer code
    that consumes :class:`exploration.novelty.NoveltyComponents`.
    """

    latent_distance: Tensor
    ensemble_variance: Tensor
    bonus_scale: Tensor
    combined_score: Tensor
    normalized_score: Tensor

    @property
    def pixel_error(self) -> Tensor:
        """Return the latent distance under the generic novelty name.

        Input: This components object.
        Output: Latent distance tensor ``[B]``.
        Mathematical meaning: Exposes ``||E(s_t)-E(s_{t+1})||_p`` where the
            reconstruction pipeline would report its pixel term.
        """
        return self.latent_distance

    @property
    def feature_error(self) -> Tensor:
        """Return the clamped scaling factor under the generic novelty name.

        Input: This components object.
        Output: Scaling tensor ``[B]``.
        Mathematical meaning: Exposes ``min(max(zeta(r),1),M)`` where the
            reconstruction pipeline would report its feature term.
        """
        return self.bonus_scale


class MetricIntrinsicReward:
    """Compute the BiGAN-latent, EME-scaled exploration bonus ``b_t``.

    Args:
        encoder: BiGAN encoder used as the metric embedding.
        ensemble: Optional ensemble supplying ``zeta(r)``. When ``None`` the
            scaling factor is identically one, which yields the latent-only
            ablation.
        max_reward_scaling: Upper clamp ``M`` on the scaling factor.
        min_reward_scaling: Lower clamp, one in the EME formulation, so the
            bonus is never shrunk below the pure metric distance.
        norm: ``"L1"`` or ``"L2"`` latent norm.
        normalize: Whether to map the raw bonus onto the extrinsic-reward scale
            with Adventurer's Eq. (5) running normalization. Keeping this on
            makes the metric bonus directly comparable with the baseline's
            intrinsic reward and keeps ``beta`` meaningful across variants.
        normalization_epsilon: Stability constant of the running normalizer.
        normalization_clip: Optional symmetric bound applied to the normalized
            bonus. Latent distances are far more concentrated than
            reconstruction errors, so early rollouts can produce a very small
            ``sigma(b)`` and an exploding Eq. (5) ratio; the default bound of
            five keeps the intrinsic stream on the same order as the extrinsic
            one without changing the ranking of transitions.
        clip_value: Optional symmetric clip applied to the final bonus.
        device: Device used for encoding and statistics.
    """

    def __init__(
        self,
        encoder,
        ensemble: Optional[EnsembleRewardVariance] = None,
        max_reward_scaling: float = 5.0,
        min_reward_scaling: float = 1.0,
        norm: str = "L2",
        normalize: bool = True,
        normalization_epsilon: float = 1.0e-8,
        normalization_clip: Optional[float] = 5.0,
        clip_value: Optional[float] = None,
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Assemble the metric term, the scaling term, and the reward scaling.

        Input: Encoder, optional ensemble, clamp bounds, norm, normalization
            settings, and device.
        Output: Initialized bonus operator.
        Mathematical meaning: Defines
            ``b_t=||E(s_t)-E(s_{t+1})||_p * min(max(zeta(r),1),M)``.
        """
        if max_reward_scaling <= 0.0 or min_reward_scaling <= 0.0:
            raise ValueError("scaling bounds must be positive")
        if max_reward_scaling < min_reward_scaling:
            raise ValueError("max_reward_scaling must be at least min_reward_scaling")
        if clip_value is not None and clip_value <= 0.0:
            raise ValueError("clip_value must be positive when provided")
        self.device = torch.device(device)
        self.discrepancy = LatentStateDiscrepancy(
            encoder, norm=norm, device=self.device
        )
        self.ensemble = ensemble
        self.max_reward_scaling = float(max_reward_scaling)
        self.min_reward_scaling = float(min_reward_scaling)
        self.normalize = bool(normalize)
        self.calculator = IntrinsicRewardCalculator(clip_value)
        if normalization_clip is not None and normalization_clip <= 0.0:
            raise ValueError("normalization_clip must be positive when provided")
        self.normalized_score = NormalizedNoveltyScore(
            epsilon=normalization_epsilon,
            clip_range=normalization_clip,
            device=self.device,
        )

    def scaling_factor(self, obs_tp1: Tensor) -> tuple[Tensor, Tensor]:
        """Return the raw ensemble variance and its clamped scaling factor.

        Input: Next-observation batch ``[B, *observation_shape]``.
        Output: ``(zeta, scale)`` tensors, both shaped ``[B]``.
        Mathematical meaning: Computes ``zeta(r)=Var_k(hat r_k(s_{t+1}))`` and
            ``min(max(zeta,1),M)``; without an ensemble both terms are one.
        """
        if self.ensemble is None:
            ones = torch.ones(obs_tp1.shape[0], device=self.device)
            return ones.clone(), ones
        variance = self.ensemble.get_variance(obs_tp1).to(self.device).float()
        scale = variance.clamp(min=self.min_reward_scaling, max=self.max_reward_scaling)
        return variance, scale

    @torch.no_grad()
    def __call__(
        self,
        obs_t: Tensor,
        obs_tp1: Tensor,
        extrinsic_reward: Optional[Tensor] = None,
        update_statistics: bool = True,
    ) -> MetricNoveltyComponents:
        """Compute the exploration bonus for one batch of transitions.

        Input: Consecutive observation batches, the matching extrinsic reward
            used to centre Eq. (5), and a statistics-update flag.
        Output: ``MetricNoveltyComponents`` whose ``normalized_score`` is the
            intrinsic reward handed to PPO.
        Mathematical meaning: Evaluates ``b_t`` and, when normalization is on,
            ``(b_t-mu(b)+mu(r^e))/sigma(b)``.
        """
        obs_t = obs_t.to(self.device)
        obs_tp1 = obs_tp1.to(self.device)
        latent_distance = self.discrepancy(obs_t, obs_tp1)
        variance, scale = self.scaling_factor(obs_tp1)
        bonus = latent_distance * scale
        if self.normalize:
            normalized = self.normalized_score(
                bonus,
                update=update_statistics,
                extrinsic_reward=extrinsic_reward,
            )
        else:
            normalized = bonus
        return MetricNoveltyComponents(
            latent_distance=latent_distance,
            ensemble_variance=variance,
            bonus_scale=scale,
            combined_score=bonus,
            normalized_score=self.calculator(normalized),
        )
