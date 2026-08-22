"""EME-style state discrepancy measured in the BiGAN latent metric space.

Adventurer scores a single state by how badly the BiGAN reconstructs it. The
metric-based exploration bonus of this module instead scores a *transition* by
how far it moves in the encoder's latent metric space,

.. math:: d_t = \\lVert E_\\psi(s_t) - E_\\psi(s_{t+1}) \\rVert_p ,

with :math:`p \\in \\{1, 2\\}`. The encoder is used purely as a learned metric:
no generator reconstruction and no discriminator pass are required, so the
bonus costs one extra encoder forward per environment step.

The full exploration bonus of Contribution 2 is

.. math:: b_t = d_t \\cdot \\min(\\max(\\zeta(r), 1), M),

where the diversity-enhanced scaling factor :math:`\\zeta(r)` is supplied by
:mod:`exploration.ensemble_scaling`. This module implements only the metric
term ``d_t``; the product is assembled in
:class:`exploration.intrinsic_reward.MetricIntrinsicReward`.
"""

from __future__ import annotations

from typing import Optional, Union

import torch
from torch import Tensor

from bigan.encoder import BiGANEncoder

_NORM_ORDERS = {"L1": 1.0, "L2": 2.0}


class LatentStateDiscrepancy:
    """Measure consecutive-state distance in the BiGAN latent space.

    Args:
        encoder: BiGAN encoder ``E_psi`` used as a frozen metric embedding.
        latent_dim: Expected latent dimension. When omitted the encoder's own
            ``latent_dim`` is used; when supplied it is validated so that a
            mismatched checkpoint fails loudly.
        norm: ``"L1"`` or ``"L2"`` latent norm.
        device: Device on which encoding and distance computation happen.
    """

    def __init__(
        self,
        encoder: BiGANEncoder,
        latent_dim: Optional[int] = None,
        norm: str = "L2",
        device: Union[torch.device, str] = "cpu",
    ) -> None:
        """Bind the encoder that defines the exploration metric.

        Input: BiGAN encoder, optional expected latent dimension, norm name,
            and device.
        Output: Initialized discrepancy operator.
        Mathematical meaning: Fixes the metric space ``(Z, ||.||_p)`` in which
            transition novelty is measured.
        """
        if norm not in _NORM_ORDERS:
            raise ValueError("norm must be 'L1' or 'L2'")
        if latent_dim is not None and latent_dim != encoder.latent_dim:
            raise ValueError("latent_dim does not match the encoder latent dimension")
        self.encoder = encoder.to(device)
        self.latent_dim = int(encoder.latent_dim)
        self.norm = norm
        self.norm_order = _NORM_ORDERS[norm]
        self.device = torch.device(device)

    def freeze(self) -> None:
        """Disable gradients for the encoder so the metric stops drifting.

        Input: This discrepancy operator.
        Output: No value; encoder parameters are set to ``requires_grad=False``
            and the module is placed in evaluation mode.
        Mathematical meaning: Fixes ``E_psi`` after BiGAN pretraining so the
            latent metric, and therefore the scale of ``d_t``, is stationary.
        """
        self.encoder.eval()
        for parameter in self.encoder.parameters():
            parameter.requires_grad_(False)

    @torch.no_grad()
    def encode(self, observations: Tensor) -> Tensor:
        """Encode a batch of observations into latent codes.

        Input: Observation batch ``[B, *observation_shape]``.
        Output: Latent tensor ``[B, latent_dim]``.
        Mathematical meaning: Computes ``z=E_psi(s)`` without building a
            gradient graph, because the bonus is an RL signal and never an
            encoder optimization path.
        """
        if observations.ndim < 2:
            raise ValueError("observations must include a batch dimension")
        return self.encoder(observations.to(self.device).float())

    @torch.no_grad()
    def __call__(self, obs_t: Tensor, obs_tp1: Tensor) -> Tensor:
        """Return the latent distance travelled by each transition.

        Input: Consecutive observation batches ``s_t`` and ``s_{t+1}``, both
            shaped ``[B, *observation_shape]``.
        Output: Non-negative tensor ``[B]`` of latent distances.
        Mathematical meaning: Computes ``d_t=||E(s_t)-E(s_{t+1})||_p``, the
            metric-based novelty of one transition.
        """
        if obs_t.shape != obs_tp1.shape:
            raise ValueError("obs_t and obs_tp1 must have identical shapes")
        latent_t = self.encode(obs_t)
        latent_tp1 = self.encode(obs_tp1)
        return (latent_t - latent_tp1).norm(p=self.norm_order, dim=-1)
