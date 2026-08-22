"""Intrinsic-reward conversion without an extra scaling coefficient.

Equation (5) already places normalized novelty on the extrinsic-reward scale.
The single beta coefficient is applied later to the intrinsic advantage inside
PPO, never here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from torch import Tensor

from exploration.novelty import NoveltyComponents


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

    def __call__(self, extrinsic_reward: Tensor, novelty: NoveltyComponents) -> IntrinsicRewardResult:
        """Return independent reward streams.

        Input: Environment reward and novelty result with matching batch shape.
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
