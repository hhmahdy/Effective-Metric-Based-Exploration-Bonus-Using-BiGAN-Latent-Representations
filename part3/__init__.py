"""Part-3 representation-intervention experiment.

This package implements the causal-attribution study that holds the
environment, PPO, forward model, reward processing, budget, and seeds fixed and
varies only the observation representation ``E`` feeding the transition
novelty ``N_T(s_t,a_t,s_{t+1}) = ||f(E(s_t),a_t) - E(s_{t+1})||_2``.
"""

from __future__ import annotations

__all__ = ["trainer"]
