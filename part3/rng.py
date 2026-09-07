"""Dedicated random-number streams for the Part-3 intervention.

Part 3 isolates the representation ``E`` as the only intervention. To keep the
comparison clean the experiment therefore requires *controlled* RNG streams for
each stochastic process rather than a single global stream. This module names
those streams and provides two primitives::

    * ``make_generator`` builds an isolated ``torch.Generator`` for a stream.
    * ``run_under_rng`` executes a callable under a forked RNG scope seeded from
      that stream.

The streams are separated so that:

* PPO/actor/critic initialization,
* representation initialization,
* representation training,
* forward-model initialization,
* forward-model training,
* replay sampling,
* BiGAN latent sampling, and
* RND target initialization

each consume their own randomness. Crucially, this does **not** claim that
trajectories remain bit-identical across representations. The requirement is:

    "Same seed and controlled RNG initialization before behavioral divergence."

Once the different representations produce different intrinsic rewards, their
trajectories are expected to diverge. That divergence is the experimental
effect, not a confounder.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TypeVar, Union

import torch

from utils.seed import create_torch_generator, derive_seed

T = TypeVar("T")

# Named stream identifiers. Each maps to a derived seed via ``derive_seed`` so
# changing the experiment seed moves every stream but preserves their mutual
# isolation.
RNG_PPO_INIT = 0x01
RNG_ACTOR_CRITIC_INIT = 0x02
RNG_REPRESENTATION_INIT = 0x03
RNG_REPRESENTATION_TRAIN = 0x04
RNG_FORWARD_MODEL_INIT = 0x05
RNG_FORWARD_TRAIN = 0x06
RNG_REPLAY_SAMPLING = 0x07
RNG_BIGAN_LATENT = 0x08
RNG_RND_TARGET = 0x09


def _cpu_device() -> torch.device:
    return torch.device("cpu")


def make_generator(
    base_seed: int,
    stream_id: int,
    device: Union[torch.device, str] = "cpu",
) -> torch.Generator:
    """Return a generator for one isolated Part-3 random stream.

    Input: Experiment base seed, a named stream id, and an optional device.
    Output: A seeded ``torch.Generator`` on the requested device.
    Mathematical meaning: Derives a stable child seed from the experiment seed
        and the stream id so each stochastic process has its own randomness.
    """
    return create_torch_generator(derive_seed(base_seed, stream_id), device)


def run_under_rng(
    base_seed: int,
    stream_id: int,
    fn: Callable[[], T],
) -> T:
    """Run ``fn`` with the global RNG scoped to a dedicated stream.

    Input: Experiment base seed, a stream id, and a zero-argument callable.
    Output: The value returned by ``fn``.
    Mathematical meaning: Temporarily replaces the active CPU (and, when
        available, CUDA) RNG state with the stream's state so that operations
        that draw from the implicit global RNG -- for example ``nn.init`` inside
        a network constructor or the implicit sampling inside the legacy BiGAN
        trainer -- are reproducible under this one stream.
    """
    generator = make_generator(base_seed, stream_id, _cpu_device())
    cpu_state = generator.get_state()
    cuda_states: list = []
    if torch.cuda.is_available():
        cuda_states = [generator.get_state() for _ in range(torch.cuda.device_count())]
    with torch.random.fork_rng():
        torch.random.set_rng_state(cpu_state)
        if cuda_states:
            torch.cuda.set_rng_state_all(cuda_states)
        return fn()
