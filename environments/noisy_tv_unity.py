"""Adapter for the original Unity Noisy-TV build of ``luchris429/noisy-tv-env``.

Why this module exists
----------------------
The reference implementation of the Noisy-TV maze is a Unity binary released by
`luchris429/noisy-tv-env <https://github.com/luchris429/noisy-tv-env>`_ ("The
Noisy TV Environment from *Large-Scale Study of Curiosity-Driven Learning*",
ICLR 2019). That repository ships

* the Unity project (``Assets/``), and
* the python client it was written against, an early ML-Agents release that is
  vendored in the repository as the ``unityagents`` package,

but the executable itself is distributed out of band (the repository's README
links a Google Drive folder, and Unity builds are not committed to git).

This module bridges that binary to the training pipeline: it wraps the vendored
:class:`unityagents.UnityEnvironment` client in a Gymnasium-style adapter with
exactly the observation/action/reward contract of
:mod:`environments.noisy_tv_maze` (84x84 RGB frames, six discrete actions,
``+1`` within ``2.5`` units of the goal). Both implementations can therefore be
used interchangeably with ``--env`` / ``ENV_NAME``.

Getting the binary
------------------
1. Download the Linux build of ``tv_maze`` from the link in that repository's
   README (the folder holds one build per platform).
2. Put the executable next to the repository or point at it explicitly, for
   example ``NoisyTVUnityConfig(file_name="/opt/tv_maze/tv_maze")``.
3. Install the client package that ships with that repository:
   ``python -m pip install -e path/to/noisy-tv-env`` (the ``unityagents``
   directory is importable as-is).

If either piece is missing, :func:`make_noisy_tv_environment` falls back to the
pure-python reconstruction, so the pipeline never depends on a Unity install;
that fallback is explicit in the returned object's ``backend`` attribute.

Upstream interface notes
------------------------
The client talks over a socket to the player and expects numpy-style dtypes that
NumPy 2 removed (``np.float_``, ``np.int_``). :func:`_ensure_numpy_aliases`
restores those aliases for the duration of the import, which is the minimal
compatibility fix needed to run that client on a modern stack. The client also
requires ``Pillow`` because observations arrive as JPEG bytes; the adapter
converts them to the pipeline's ``uint8`` ``(H, W, 3)`` layout.
"""

from __future__ import annotations

import glob
import os
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Tuple

import numpy as np

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError as error:  # pragma: no cover - gymnasium is a hard dependency
    raise ImportError(
        "NoisyTVUnity requires gymnasium; install it with "
        "`python -m pip install -r requirements-master.txt`"
    ) from error

from environments.noisy_tv_maze import (
    ENVIRONMENT_ID as MAZE_ENVIRONMENT_ID,
    GOAL_REACH_DISTANCE,
    NoisyTVMazeEnv,
)


ENVIRONMENT_ID = "NoisyTVUnity-v0"
"""Gymnasium registration ID of the adapter around the original Unity build."""

ENTRY_POINT = "environments.noisy_tv_unity:make_unity_environment"
"""Import path Gymnasium resolves for :data:`ENVIRONMENT_ID`."""

DEFAULT_FILE_NAME = "tv_maze"
"""Upstream executable name, as used by the repository's ``tv_test.py``."""

ACTIONS = 6
"""Discrete actions of the upstream brain (``actionSize: 6``)."""

OBSERVATION_SHAPE = (84, 84, 3)
"""Upstream camera resolution (``84x84``, ``blackAndWhite: 0``)."""


@dataclass
class NoisyTVUnityConfig:
    """Connection and reset parameters for the Unity player.

    Attributes:
        file_name: Path to the executable without its platform suffix, or the
            directory containing it (``tv_test.py`` uses the bare ``"tv_maze"``).
        worker_id: Upstream worker id selecting the port offset.
        base_port: Upstream base port of the socket connection.
        start_loc: Upstream ``startLoc`` reset parameter.
        door: Upstream ``door`` reset parameter.
        tv: Upstream ``tv`` reset parameter.
        train_mode: Upstream ``train_mode`` flag of ``env.reset``.
        timeout: Seconds to wait for the player to answer.
    """

    file_name: str = DEFAULT_FILE_NAME
    worker_id: int = 0
    base_port: int = 5005
    start_loc: float = 0.0
    door: float = 1.0
    tv: float = 1.0
    train_mode: bool = True
    timeout: float = 60.0
    extra_reset_parameters: Dict[str, float] = field(default_factory=dict)

    def reset_parameters(self) -> Dict[str, float]:
        """Return the reset dictionary the upstream player expects.

        Input: This configuration.
        Output: Mapping with ``startLoc``, ``door``, and ``tv`` plus any extras.
        Mathematical meaning: Selects the upstream episode condition (start
            pose and door/television behaviour).
        """
        parameters = {"startLoc": float(self.start_loc), "door": float(self.door), "tv": float(self.tv)}
        parameters.update({str(key): float(value) for key, value in self.extra_reset_parameters.items()})
        return parameters


def _ensure_numpy_aliases() -> None:
    """Restore NumPy 1.x dtype aliases used by the vendored client.

    Input: None.
    Output: No value; ``np.float_``/``np.int_`` exist afterwards if removed.
    Mathematical meaning: None; it is a compatibility shim for the upstream
        client, which was written against NumPy 1.x.
    """
    if not hasattr(np, "float_"):
        np.float_ = np.float64  # type: ignore[attr-defined]
    if not hasattr(np, "int_"):
        np.int_ = np.int64  # type: ignore[attr-defined]


def _client_module() -> Any:
    """Import the upstream python client with its NumPy aliases in place.

    Input: None.
    Output: The ``unityagents`` module object, or ``None`` when unavailable.
    Mathematical meaning: None; this is the transport layer to the Unity build.
    """
    _ensure_numpy_aliases()
    try:
        import unityagents
    except ImportError:
        return None
    return unityagents


def unity_binary_available(file_name: str = DEFAULT_FILE_NAME) -> bool:
    """Check whether a Unity player matching ``file_name`` can be launched.

    Input: Executable path without suffix, or a directory to search.
    Output: ``True`` when an executable candidate exists for this platform.
    Mathematical meaning: None; it decides whether the original build or the
        reconstruction is used.
    """
    directory = file_name if os.path.isdir(file_name) else os.path.dirname(os.path.abspath(file_name))
    name = os.path.basename(file_name.rstrip("/")) or DEFAULT_FILE_NAME
    if not os.path.isdir(directory):
        return False
    if os.path.isfile(file_name) or os.path.isfile(file_name + ".exe"):
        return True
    patterns = [
        f"{name}.x86_64",
        f"{name}_x86_64",
        f"{name}.x86",
        f"{name}.exe",
        f"{name}.app",
    ]
    for pattern in patterns:
        if glob.glob(os.path.join(directory, pattern)):
            return True
    # A build folder may nest the executable one level down (e.g. tv_maze_Data).
    for pattern in patterns:
        if glob.glob(os.path.join(directory, "*", pattern)):
            return True
    return False


def unity_available(config: Optional[NoisyTVUnityConfig] = None) -> bool:
    """Check whether the original Unity build can be created right now.

    Input: Optional configuration naming the executable.
    Output: ``True`` when the client package imports and an executable exists.
    Mathematical meaning: None; it gates the ``"auto"`` backend choice.
    """
    configuration = config or NoisyTVUnityConfig()
    if _client_module() is None:
        return False
    return unity_binary_available(configuration.file_name)


class UnityEnvironmentAdapter:
    """Gymnasium-style adapter around the upstream ``unityagents`` client.

    Args:
        config: Connection and reset parameters of the player.
        client: Pre-built client, used by tests; when omitted the adapter
            constructs :class:`unityagents.UnityEnvironment` itself.

    The adapter reproduces the contract of :class:`NoisyTVMazeEnv`: ``reset``
    returns ``(observation, info)``, ``step`` returns
    ``(observation, reward, terminated, truncated, info)`` with ``uint8`` RGB
    frames shaped :data:`OBSERVATION_SHAPE`, and ``info`` always reports
    ``distance_to_goal`` and ``distance_to_television``. The distance to the
    goal is *reconstructed* from the upstream state ``(x, z)`` because the
    upstream brain does not expose it; with the upstream goal at
    ``(-10, 60)`` this reproduces the reward criterion exactly.
    """

    def __init__(
        self,
        config: Optional[NoisyTVUnityConfig] = None,
        client: Any = None,
    ) -> None:
        """Connect to (or adopt) a player and read its brain parameters.

        Input: Optional configuration and optional pre-built client.
        Output: A connected adapter with ``backend = "unity"``.
        Mathematical meaning: Establishes the interface through which rollouts
            sample transitions from the upstream MDP.
        """
        self.config = config or NoisyTVUnityConfig()
        self.backend = "unity"
        if client is None:
            module = _client_module()
            if module is None:
                raise RuntimeError(
                    "the upstream `unityagents` client is not importable; install it with "
                    "`python -m pip install -e path/to/noisy-tv-env` (the package is vendored "
                    "in that repository) or use the pure-python NoisyTVMaze-v0 instead"
                )
            if not unity_binary_available(self.config.file_name):
                raise RuntimeError(
                    f"no Unity player matching {self.config.file_name!r} was found; download the "
                    "build linked in the README of https://github.com/luchris429/noisy-tv-env "
                    "and pass its path through NoisyTVUnityConfig(file_name=...)"
                )
            client = module.UnityEnvironment(
                file_name=self.config.file_name,
                worker_id=self.config.worker_id,
                base_port=self.config.base_port,
            )
        self.client = client
        self.brain_name = self.client.brain_names[0]
        self.brain = self.client.brains[self.brain_name]
        if self.brain.action_space_type != "discrete":
            raise RuntimeError("the upstream Noisy-TV brain is expected to be discrete")
        self.action_space = spaces.Discrete(int(self.brain.action_space_size))
        self.observation_space = spaces.Box(
            low=0, high=255, shape=OBSERVATION_SHAPE, dtype=np.uint8
        )
        self._last_info: Dict[str, Any] = {}
        self._last_frame: Optional[np.ndarray] = None
        self._needs_reset = True

    # ------------------------------------------------------------- helpers
    def _frame(self, brain_info: Any) -> np.ndarray:
        """Extract the ``uint8`` RGB frame from an upstream ``BrainInfo``.

        Input: Upstream ``BrainInfo`` for one agent.
        Output: ``uint8`` array shaped :data:`OBSERVATION_SHAPE`.
        Mathematical meaning: Converts the player's rendered pixels into the
            observation the pipeline consumes.
        """
        observations = brain_info.observations[0]
        frame = np.asarray(observations[0])
        if frame.dtype != np.uint8:
            # The upstream client divides by 255 and may grayscale; restore the
            # documented 0-255 RGB layout.
            if frame.ndim == 3 and frame.shape[-1] == 1:
                frame = np.repeat(frame, 3, axis=-1)
            frame = np.clip(np.asarray(frame) * 255.0, 0, 255).astype(np.uint8)
        if frame.ndim == 3 and frame.shape[0] == 3 and frame.shape[-1] != 3:
            frame = np.transpose(frame, (1, 2, 0))
        return frame

    def _info(self, brain_info: Any) -> Dict[str, Any]:
        """Translate upstream state/reward fields into the pipeline's info dict.

        Input: Upstream ``BrainInfo`` for one agent.
        Output: Dictionary with the pose, distances, and reward diagnostics.
        Mathematical meaning: The upstream brain exposes only ``(x, z)``; the
            distance to the goal sphere at ``(-10, 60)`` is recovered from it,
            which is the quantity the ``+1`` reward is defined by.
        """
        state = np.asarray(brain_info.states[0]).reshape(-1)
        position_x = float(state[0]) if state.size > 0 else float("nan")
        position_z = float(state[1]) if state.size > 1 else float("nan")
        goal_x, goal_z = -10.0, 60.0
        television_x, television_z = 45.02, 39.97
        return {
            "agent_position": (position_x, position_z),
            "distance_to_goal": float(np.hypot(position_x - goal_x, position_z - goal_z)),
            "distance_to_television": float(
                np.hypot(position_x - television_x, position_z - television_z)
            ),
            "reward": float(np.asarray(brain_info.rewards).reshape(-1)[0]),
            "upstream_state": tuple(float(value) for value in state),
        }

    # ----------------------------------------------------------- Gymnasium
    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Reset the player with the configured upstream reset parameters.

        Input: Optional seed (the upstream client has no seeding API; the
            player's own randomness is used) and an options dictionary that may
            override ``start_loc``, ``door``, and ``tv`` for this episode.
        Output: ``(observation, info)`` for the initial state.
        Mathematical meaning: Samples the initial state from the upstream
            initial-state distribution described by the reset parameters.
        """
        options = options or {}
        parameters = self.config.reset_parameters()
        for key, option in (
            ("startLoc", options.get("start_loc")),
            ("door", options.get("door")),
            ("tv", options.get("tv")),
        ):
            if option is not None and not isinstance(option, str):
                parameters[key] = float(option)
        brain_info = self.client.reset(
            train_mode=self.config.train_mode, config=parameters
        )[self.brain_name]
        self._needs_reset = False
        info = self._info(brain_info)
        self._last_info = info
        self._last_frame = self._frame(brain_info)
        return self._last_frame, info

    def step(self, action: Any) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """Send one discrete action to the player and normalize the result.

        Input: Discrete action in ``0..5``: ``0`` no-op, ``1`` forward,
            ``2``/``3`` rotate, ``4`` press the door, ``5`` change the channel.
        Output: ``(observation, reward, terminated, truncated, info)``.
        Mathematical meaning: Executes one transition of the upstream MDP. The
            upstream player has no truncation concept beyond its own episode
            limits, so ``truncated`` is ``False`` and ``terminated`` carries the
            global-done flag.
        """
        if self._needs_reset:
            raise RuntimeError("environment must be reset before step")
        action_value = int(np.asarray(action).reshape(-1)[0])
        if not self.action_space.contains(action_value):
            raise ValueError(f"action must be an element of {self.action_space}, got {action!r}")
        brain_info = self.client.step([action_value])[self.brain_name]
        done = bool(np.any(np.asarray(brain_info.local_done)))
        global_done = bool(getattr(self.client, "global_done", False))
        self._needs_reset = done or global_done
        info = self._info(brain_info)
        self._last_info = info
        self._last_frame = self._frame(brain_info)
        return self._last_frame, float(info["reward"]), done or global_done, False, info

    def render(self) -> Optional[np.ndarray]:
        """Return the latest frame without stepping.

        Input: This adapter.
        Output: The most recent observation, or ``None`` before the first reset.
        Mathematical meaning: Reads the current observation the player sent.
        """
        return None if self._last_frame is None else self._last_frame.copy()

    def close(self) -> None:
        """Close the socket connection and terminate the player.

        Input: This adapter.
        Output: No value.
        Mathematical meaning: Ends interaction with the upstream MDP instance.
        """
        close = getattr(self.client, "close", None)
        if callable(close):
            close()

    def __enter__(self) -> "UnityEnvironmentAdapter":
        """Enter a context manager and return this adapter.

        Input: This adapter.
        Output: The same adapter instance.
        Mathematical meaning: Bounds the lifetime of the player process.
        """
        return self

    def __exit__(self, exc_type: object, exc_value: object, traceback: object) -> None:
        """Close the player when leaving a context-manager block.

        Input: Standard context-manager exception information.
        Output: No value; the player is closed.
        Mathematical meaning: Guarantees the external process is released.
        """
        self.close()


def make_unity_environment(**kwargs: Any) -> UnityEnvironmentAdapter:
    """Gymnasium entry point for :data:`ENVIRONMENT_ID`.

    Input: Keyword arguments forwarded to :class:`NoisyTVUnityConfig`
        (``file_name``, ``worker_id``, ``start_loc``, ``door``, ``tv``, ...).
    Output: A connected :class:`UnityEnvironmentAdapter`.
    Mathematical meaning: Instantiates the upstream MDP through the registered
        environment ID.
    """
    config_fields = set(NoisyTVUnityConfig().__dataclass_fields__)
    config_kwargs = {key: value for key, value in kwargs.items() if key in config_fields}
    configuration = NoisyTVUnityConfig(**config_kwargs)
    return UnityEnvironmentAdapter(configuration)


def register_environment(environment_id: str = ENVIRONMENT_ID) -> str:
    """Register the Unity adapter and the pure-python maze with Gymnasium.

    Input: Registration ID of the Unity adapter.
    Output: The registered Unity adapter ID.
    Mathematical meaning: Makes both implementations of the Noisy-TV MDP
        addressable by name, so ``--env NoisyTVUnity-v0`` and
        ``--env NoisyTVMaze-v0`` are interchangeable.
    """
    registry = getattr(gym, "registry", None) or gym.envs.registry
    if MAZE_ENVIRONMENT_ID not in registry:
        gym.register(
            id=MAZE_ENVIRONMENT_ID,
            entry_point="environments.noisy_tv_maze:NoisyTVMazeEnv",
            reward_threshold=None,
            disable_env_checker=False,
        )
    if environment_id not in registry:
        gym.register(
            id=environment_id,
            entry_point=ENTRY_POINT,
            reward_threshold=None,
            disable_env_checker=False,
        )
    return environment_id


def make_noisy_tv_environment(
    backend: str = "auto",
    **backend_kwargs: Any,
):
    """Create the Noisy-TV maze from either the Unity build or the reconstruction.

    Input: ``backend`` -- ``"auto"`` (Unity when the client and the executable
        are present, otherwise the reconstruction), ``"unity"`` (require the
        original build), or ``"python"`` (always the reconstruction) -- plus
        keyword arguments forwarded to the chosen implementation.
    Output: ``NoisyTVMazeEnv`` or ``UnityEnvironmentAdapter``.
    Mathematical meaning: Same MDP family, two implementations; the choice is
        recorded on the returned object's ``backend`` attribute.

    Raises:
        ValueError: If ``backend`` is not one of the three accepted values.
        RuntimeError: If ``backend="unity"`` but the client or the executable is
            missing, so that a silent fallback can never invalidate a
            comparison between the two exploratory signals.
    """
    if backend not in {"auto", "unity", "python"}:
        raise ValueError("backend must be 'auto', 'unity', or 'python'")
    if backend == "auto":
        backend = "unity" if unity_available() else "python"
    if backend == "unity":
        return make_unity_environment(**backend_kwargs)
    py_kwargs = {
        key: value
        for key, value in backend_kwargs.items()
        if key not in set(NoisyTVUnityConfig().__dataclass_fields__)
    }
    environment = NoisyTVMazeEnv(**py_kwargs)
    environment.backend = "python"
    return environment


__all__ = [
    "ENVIRONMENT_ID",
    "MAZE_ENVIRONMENT_ID",
    "GOAL_REACH_DISTANCE",
    "NoisyTVUnityConfig",
    "UnityEnvironmentAdapter",
    "make_noisy_tv_environment",
    "make_unity_environment",
    "register_environment",
    "unity_available",
    "unity_binary_available",
]
