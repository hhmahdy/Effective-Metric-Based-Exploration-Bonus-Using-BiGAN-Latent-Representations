"""CorridorTV: a PyColab-style sparse-reward gridworld for the Part-3 experiment.

**Engine note (documented substitution).** The DeepMind ``pycolab`` package is
available and importable in the target environment (see the module import
below), but this module implements the gridworld **from scratch** as a
PyColab-style engine. The reason is that the Part-3 protocol requires an exact,
auditable observation / reward / info contract that pycolab's rendering and
rollout APIs do not produce out of the box: a ``(64, 64, 3)`` upsampled RGB
observation, an action- and agent-independent noisy-TV / random-walk decoy, a
fixed controllable-state table of exactly 73 entries, and a bespoke ``info``
contract. Implementing the engine directly guarantees that contract and lets the
unit tests validate it deterministically. This substitution follows the Part-3
protocol: *"an equivalent from-scratch PyColab-style gridworld is acceptable
only if it reproduces the exact observation/reward/info contract below, and the
substitution must be documented in the module docstring."* A thin ``pycolab``
import is kept as a documentation/reference anchor and to confirm availability;
the environment does not depend on it for its behavior.

**Fixed preprocessing.** The internal gridworld renders a ``32 x 32`` cell RGB
observation. The ``Gymnasium`` observation space is
``Box(0, 255, (64, 64, 3), dtype=uint8)``: the raw ``32 x 32 x 3`` grid is
**nearest-neighbor upsampled** to ``64 x 64 x 3`` (2x in each spatial axis)
because the repository's encoder convolutional trunk (``k8/s4 -> k4/s2 -> k3/s1``)
requires at least 64 px of spatial resolution. This resize is part of the
**fixed** preprocessing and is *identical across representations and seeds*; it
is recorded in ``metadata`` and in this docstring. ``BiGANEncoder((64, 64, 3),
128, 256, 512)`` accepts the resulting observation (verified by a smoke test).

**Architecture summary.**

* **Controllable state**: agent position (winding 1-cell-wide corridor) and door
  state (open / closed), toggled only by the switch tile.
* **Uncontrollable visual factors** (action- and agent-independent): (a) a noisy
  TV / flicker region replaced by fresh per-frame i.i.d. uniform noise, and (b) a
  random-walk colored sprite doing a uniform 4-neighbor walk in a bounded region.
* **Sparse task reward**: ``+1`` exactly on entering the goal tile, after which
  the episode terminates.

**Reachable controllable configurations = 73.** An exhaustive BFS over the
deterministic transition function yields exactly 73 reachable
``(agent position, door state)`` pairs. The layout is a winding 37-cell corridor
(serpentine, offset into the grid); the door is the final corridor cell before
the goal and a toggle switch sits earlier on the corridor. This gives 36
closed-door positions (the door and the goal are blocked while closed) plus 37
open-door positions (the door cell is traversable) = 73.

**Ground-truth controllability.** ``CORRIDORTV_CONTROLLABLE_STATES = 73`` and
``corridor_env.controllable_table`` maps each reachable ``(position, door_state)``
pair to a stable index ``0..72``.

**info contract** (provided on every reset and step): ``agent_position``
``(x, y)``, ``door_open`` (bool), ``controllable_state`` (int ``0..72``),
``goal_reached`` (bool), ``flicker_active`` (bool), ``decoy_position``
``(x, y)``, and ``seed`` (int).
"""

from __future__ import annotations

import numpy as np

# Reference / availability anchor for the documented from-scratch substitution.
import pycolab  # noqa: F401  (kept to confirm availability and record the engine lineage)

import gymnasium as gym
from gymnasium import spaces
from gymnasium import wrappers


# ---------------------------------------------------------------------------
# Public constants / metadata
# ---------------------------------------------------------------------------

CORRIDORTV_CONTROLLABLE_STATES = 73
CORRIDORTV_INTERNAL_GRID = 32
CORRIDORTV_OBSERVATION_SIZE = 64
CORRIDORTV_RGB_CHANNELS = 3
CORRIDORTV_MAX_EPISODE_STEPS = 1000
CORRIDORTV_GOAL_REWARD = 1.0

# Action space: 0=up, 1=down, 2=left, 3=right. Deterministic movement.
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]  # up, down, left, right

# Number of winding corridor cells. With the door as the final cell before the
# goal and a toggle switch earlier, the BFS reaches exactly 73 controllable
# configurations (36 positions * closed door + 37 positions * open door).
CORRIDOR_LENGTH = 37
CORRIDOR_WIDTH = 8
CORRIDOR_DOOR_INDEX = 36        # final corridor cell before the goal
CORRIDOR_SWITCH_INDEX = 18      # earlier corridor cell; toggles the door

# Wall / reachable dimensions are fixed; the grid is CORRIDORTV_INTERNAL_GRID
# x CORRIDORTV_INTERNAL_GRID cells, with a 1-cell border margin.

# ---------------------------------------------------------------------------
# Palette (RGB uint8)
# ---------------------------------------------------------------------------

_COLOR_WALL = np.array([20, 20, 20], dtype=np.uint8)
_COLOR_CORRIDOR = np.array([80, 90, 110], dtype=np.uint8)
_COLOR_SWITCH = np.array([230, 180, 40], dtype=np.uint8)
_COLOR_DOOR_OPEN = np.array([60, 200, 90], dtype=np.uint8)
_COLOR_DOOR_CLOSED = np.array([190, 60, 60], dtype=np.uint8)
_COLOR_GOAL = np.array([250, 210, 70], dtype=np.uint8)
_COLOR_AGENT = np.array([240, 240, 240], dtype=np.uint8)
_COLOR_DECOY = np.array([170, 90, 220], dtype=np.uint8)


def _serpentine_path(width: int, length: int) -> list[tuple[int, int]]:
    """Build a winding 1-cell-wide corridor path as grid coordinates.

    Input: Corridor width and number of cells.
    Output: List of ``(x, y)`` grid coordinates forming a connected serpentine,
        offset by one cell from the grid border so the corridor is surrounded by
        a one-cell wall margin.
    Mathematical meaning: Defines the set of walkable corridor cells of the
        deterministic transition function.
    """
    coordinates: list[tuple[int, int]] = []
    for index in range(length):
        row = index // width
        column = index % width
        x = column if row % 2 == 0 else (width - 1 - column)
        # +1 offset keeps a uniform 1-cell wall border around the corridor.
        coordinates.append((x + 1, row + 1))
    return coordinates


def _nearest_neighbor_upsample(grid: np.ndarray, factor: int) -> np.ndarray:
    """Upsample an image by an integer factor using nearest-neighbor repeats.

    Input: ``(H, W, C)`` array and positive integer factor.
    Output: ``(H*factor, W*factor, C)`` array.
    Mathematical meaning: Implements the fixed 32x32 -> 64x64 preprocessing in
        the observation contract.
    """
    if factor <= 0:
        raise ValueError("factor must be positive")
    return np.repeat(np.repeat(grid, factor, axis=0), factor, axis=1)


class CorridorTVEnv(gym.Env):
    """PyColab-style sparse-reward gridworld with action-independent distractors.

    Args:
        max_episode_steps: Episode step budget; the environment truncates with
            ``truncated=True`` after this many steps. Defaults to 1000. The
            registered environment is behaviour-equivalent to
            ``gymnasium.wrappers.TimeLimit(env, max_episode_steps=1000)``.
    """

    metadata = {
        "render_modes": ["rgb_array"],
        "corridortv_internal_grid": CORRIDORTV_INTERNAL_GRID,
        "corridortv_observation_size": CORRIDORTV_OBSERVATION_SIZE,
        "corridortv_rgb_channels": CORRIDORTV_RGB_CHANNELS,
        "corridortv_preprocessing": (
            "nearest-neighbor 32x32x3 -> 64x64x3 upsample (fixed preprocessing, "
            "identical across representations and seeds)"
        ),
        "corridortv_controllable_states": CORRIDORTV_CONTROLLABLE_STATES,
        "corridortv_max_episode_steps": CORRIDORTV_MAX_EPISODE_STEPS,
    }

    def __init__(self, max_episode_steps: int = CORRIDORTV_MAX_EPISODE_STEPS) -> None:
        """Build the layout, spaces, and controllable-state table.

        Input: Episode step budget.
        Output: An initialized ``CorridorTVEnv``.
        Mathematical meaning: Assembles the deterministic controllable MDP
            (corridor, switch, door, goal) and the fixed observation contract.
        """
        if max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        super().__init__()
        self.max_episode_steps = int(max_episode_steps)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(
            low=0,
            high=255,
            shape=(CORRIDORTV_OBSERVATION_SIZE, CORRIDORTV_OBSERVATION_SIZE, CORRIDORTV_RGB_CHANNELS),
            dtype=np.uint8,
        )

        # Deterministic controllable layout.
        self._path = _serpentine_path(CORRIDOR_WIDTH, CORRIDOR_LENGTH)
        self._coord_to_index: dict[tuple[int, int], int] = {
            coordinates: index for index, coordinates in enumerate(self._path)
        }
        self.door_index = CORRIDOR_DOOR_INDEX
        self.switch_index = CORRIDOR_SWITCH_INDEX
        self.door_coordinate = self._path[self.door_index]
        self.start_index = 0
        self.switch_coordinate = self._path[self.switch_index]

        # The goal sits one cell directly BELOW the door, in the grid's wall
        # margin. The corridor occupies the rows above it, so the goal's only
        # walkable neighbour is the door cell: it is unreachable until the door
        # is open, which is the intended gating semantics.
        self.goal_coordinate = (
            self.door_coordinate[0],
            self.door_coordinate[1] + 1,
        )

        # Uncontrollable visual factor regions (action- and agent-independent).
        self._flicker_region = (24, 29, 22, 29)  # (y0, y1, x0, x1) inclusive; 6x8 cells
        self._decoy_region = (12, 20, 12, 24)    # (y0, y1, x0, x1) inclusive
        self._validate_regions_do_not_overlap_corridor()

        # Controllable-state table built from BFS over the transition function.
        self.controllable_table: dict[tuple[int, int], int] = self._build_controllable_table()

        self._positions = np.zeros((CORRIDORTV_INTERNAL_GRID, CORRIDORTV_INTERNAL_GRID, CORRIDORTV_RGB_CHANNELS), dtype=np.uint8)
        self._agent_index = self.start_index
        self._door_open = False
        self._goal_reached = False
        self._flicker_active = True
        self._step_count = 0
        self._last_seed: int | None = None
        self._visual_rng = np.random.RandomState(0)
        self._flicker_noise: np.ndarray | None = None
        self._decoy_position = self._decoy_region_start()
        self._terminal = False

    # -- layout helpers ----------------------------------------------------

    def _validate_regions_do_not_overlap_corridor(self) -> None:
        """Ensure flicker/decoy regions never overlap the controllable corridor.

        Input: Region bounds.
        Output: No value; raises ``ValueError`` on overlap.
        Mathematical meaning: Guarantees the visual distractors cannot mask or
            alter any controllable transition.
        """
        corridor_cells = set(self._path)
        corridor_cells.add(self.goal_coordinate)
        for y0, y1, x0, x1 in (self._flicker_region, self._decoy_region):
            for y in range(y0, y1 + 1):
                for x in range(x0, x1 + 1):
                    if (x, y) in corridor_cells:
                        raise ValueError("visual factor region overlaps the corridor")

    def _decoy_region_start(self) -> tuple[int, int]:
        """Return the top-left cell of the decoy region as a starting point."""
        y0, _y1, x0, _x1 = self._decoy_region
        return (x0, y0)

    # -- transition function ----------------------------------------------

    def _transition(self, index: int, door_open: bool, action: int) -> tuple[int, bool, bool]:
        """Return the deterministic controllable transition for one action.

        Input: Position index, door-open flag, and action ``0..3``.
        Output: ``(new_index, new_door_open, terminated)``.
        Mathematical meaning: Implements deterministic movement on the corridor;
            a closed door blocks passage, the switch toggles the door, and the
            goal terminates the episode with the sparse reward.
        """
        current = self._path[index]
        dx, dy = ACTIONS[action]
        target = (current[0] + dx, current[1] + dy)
        if target == self.goal_coordinate:
            # Only reachable through the open door, so entering the goal
            # terminates the episode.
            return index, door_open, True
        if target not in self._coord_to_index:
            # Wall: movement is blocked, state unchanged.
            return index, door_open, False
        next_index = self._coord_to_index[target]
        if next_index == self.door_index and not door_open:
            # Closed door blocks passage.
            return index, door_open, False
        next_door_open = door_open ^ (True if next_index == self.switch_index else False)
        return next_index, next_door_open, False

    def _build_controllable_table(self) -> dict[tuple[int, int], int]:
        """BFS the deterministic transition function and build the state table.

        Input: None; uses the fixed layout.
        Output: Dict mapping ``(position_index, door_open)`` to a stable index.
        Mathematical meaning: Enumerates every reachable controllable
            configuration and assigns a stable ``0..72`` index.
        """
        from collections import deque

        start = (self.start_index, False)
        reachable: set[tuple[int, int]] = {start}
        queue = deque([start])
        while queue:
            index, door_open = queue.popleft()
            for action in range(int(self.action_space.n)):
                next_index, next_door_open, terminated = self._transition(index, door_open, action)
                if terminated:
                    # Goal is a terminal controllable state; it is not added to
                    # the reachable controllable configuration count.
                    continue
                state = (next_index, next_door_open)
                if state not in reachable:
                    reachable.add(state)
                    queue.append(state)
        ordered = sorted(reachable, key=lambda state: (state[0], int(state[1])))
        if len(ordered) != CORRIDORTV_CONTROLLABLE_STATES:
            raise ValueError(
                f"CorridorTV layout must yield exactly {CORRIDORTV_CONTROLLABLE_STATES} "
                f"reachable controllable configurations, got {len(ordered)}"
            )
        return {state: index for index, state in enumerate(ordered)}

    @staticmethod
    def ground_truth_controllability() -> int:
        """Return the number of reachable controllable configurations (73).

        Input: None.
        Output: ``CORRIDORTV_CONTROLLABLE_STATES``.
        Mathematical meaning: The auditable ground truth the BFS test checks.
        """
        return CORRIDORTV_CONTROLLABLE_STATES

    # -- RNG / visual factors ---------------------------------------------

    def _reset_visual_rng(self, seed: int) -> None:
        """Seed the action-independent visual RNG.

        Input: Non-negative episode seed.
        Output: No value; the visual stream is reset.
        Mathematical meaning: Distractors are deterministic given the seed and
            independent of the agent's actions.
        """
        self._visual_rng = np.random.RandomState(int(seed) & 0xFFFFFFFF)

    def _draw_flicker(self) -> None:
        """Replace the noisy-TV region with fresh per-frame uniform noise.

        Input: None; uses the visual RNG.
        Output: No value; ``self._flicker_noise`` is updated in place.
        Mathematical meaning: Implements the i.i.d. per-frame noisy-TV factor.
        """
        y0, y1, x0, x1 = self._flicker_region
        height = y1 - y0 + 1
        width = x1 - x0 + 1
        self._flicker_noise = self._visual_rng.randint(
            0, 256, size=(height, width, CORRIDORTV_RGB_CHANNELS), dtype=np.uint8
        )

    def _sample_decoy_position(self) -> None:
        """Place the decoy sprite at a random initial position in its region.

        Input: None; uses the visual RNG.
        Output: No value; ``self._decoy_position`` is set.
        Mathematical meaning: Initialises the random-walk visual decoy.
        """
        y0, y1, x0, x1 = self._decoy_region
        self._decoy_position = (
            int(self._visual_rng.randint(x0, x1 + 1)),
            int(self._visual_rng.randint(y0, y1 + 1)),
        )

    def _advance_decoy(self) -> None:
        """Move the decoy one uniform 4-neighbor step inside its region.

        Input: None; uses the visual RNG and current decoy position.
        Output: No value; ``self._decoy_position`` advances.
        Mathematical meaning: Random-walk visual factor that never affects the
            controllable state.
        """
        y0, y1, x0, x1 = self._decoy_region
        cx, cy = self._decoy_position
        candidates: list[tuple[int, int]] = []
        for dx, dy in ACTIONS:
            nx, ny = cx + dx, cy + dy
            if x0 <= nx <= x1 and y0 <= ny <= y1:
                candidates.append((nx, ny))
        if not candidates:
            return
        pick = int(self._visual_rng.randint(0, len(candidates)))
        self._decoy_position = candidates[pick]

    # -- rendering ---------------------------------------------------------

    def _render_rgb(self) -> np.ndarray:
        """Render the current state to a ``(64, 64, 3)`` uint8 observation.

        Input: Current controllable + uncontrollable state.
        Output: ``uint8`` array shaped ``(64, 64, 3)``.
        Mathematical meaning: Composes the fixed preprocessing (32x32 -> 64x64).
        """
        grid = np.tile(_COLOR_WALL, (CORRIDORTV_INTERNAL_GRID, CORRIDORTV_INTERNAL_GRID, 1))
        for coordinate in self._path:
            x, y = coordinate
            grid[y, x] = _COLOR_CORRIDOR
        sx, sy = self.switch_coordinate
        grid[sy, sx] = _COLOR_SWITCH
        dx, dy = self.door_coordinate
        grid[dy, dx] = _COLOR_DOOR_OPEN if self._door_open else _COLOR_DOOR_CLOSED
        gx, gy = self.goal_coordinate
        grid[gy, gx] = _COLOR_GOAL
        ax, ay = self._path[self._agent_index]
        grid[ay, ax] = _COLOR_AGENT
        # Noisy-TV flicker (uncontrollable; overrides wall pixels only).
        if self._flicker_noise is not None:
            y0, y1, x0, x1 = self._flicker_region
            grid[y0:y1 + 1, x0:x1 + 1] = self._flicker_noise
        # Random-walk decoy (uncontrollable).
        dex, dey = self._decoy_position
        grid[dey, dex] = _COLOR_DECOY
        return _nearest_neighbor_upsample(grid, 2)

    def render(self) -> np.ndarray:
        """Return the current observation as an ``rgb_array``.

        Input: No arguments.
        Output: ``(64, 64, 3)`` uint8 array.
        Mathematical meaning: Exposes the fixed-preprocessing observation for
            visualization / inspection.
        """
        return self._render_rgb()

    # -- gymnasium API -----------------------------------------------------

    def _make_info(self) -> dict:
        """Build the Part-3 info contract.

        Input: Internal state.
        Output: Dict with ``agent_position``, ``door_open``,
            ``controllable_state``, ``goal_reached``, ``flicker_active``,
            ``decoy_position``, and ``seed``.
        Mathematical meaning: Exposes the controllable state and the
            action-independent factors for Part-3 diagnostics.
        """
        return {
            "agent_position": tuple(self._path[self._agent_index]),
            "door_open": bool(self._door_open),
            "controllable_state": self.controllable_table[(self._agent_index, bool(self._door_open))],
            "goal_reached": bool(self._goal_reached),
            "flicker_active": bool(self._flicker_active),
            "decoy_position": tuple(self._decoy_position),
            "seed": int(self._last_seed) if self._last_seed is not None else 0,
        }

    def reset(self, *, seed: int | None = None, options: dict | None = None) -> tuple[np.ndarray, dict]:
        """Reset the episode and draw the initial uncontrollable factors.

        Input: Optional episode seed and options.
        Output: ``(observation, info)`` with a ``(64, 64, 3)`` uint8 observation.
        Mathematical meaning: Resets the deterministic controllable state and
            re-seeds the action-independent visual stream.
        """
        super().reset(seed=seed)
        self._last_seed = int(seed) if seed is not None else 0
        self._reset_visual_rng(self._last_seed)
        self._agent_index = self.start_index
        self._door_open = False
        self._goal_reached = False
        self._step_count = 0
        self._terminal = False
        self._flicker_active = True
        self._draw_flicker()
        self._sample_decoy_position()
        observation = self._render_rgb()
        return observation, self._make_info()

    def step(self, action: int) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Advance the world by one deterministic action.

        Input: Integer action ``0..3``.
        Output: ``(observation, reward, terminated, truncated, info)``.
        Mathematical meaning: Executes the controllable transition, applies the
            sparse goal reward, truncates at ``max_episode_steps``, and advances
            the action-independent visual factors.
        """
        if not 0 <= int(action) < int(self.action_space.n):
            raise ValueError(f"action must be in [0, {int(self.action_space.n)}), got {action}")
        if self._terminal:
            raise RuntimeError("step called on a terminated CorridorTV episode; reset first")

        next_index, next_door_open, terminated = self._transition(
            self._agent_index, self._door_open, int(action)
        )
        self._agent_index = next_index
        self._door_open = next_door_open
        self._goal_reached = bool(terminated)
        reward = CORRIDORTV_GOAL_REWARD if terminated else 0.0

        # Advance the action-independent visual factors (drawn every frame).
        self._draw_flicker()
        self._advance_decoy()
        self._step_count += 1

        truncated = bool(self._step_count >= self.max_episode_steps)
        if terminated:
            self._terminal = True
        elif truncated:
            self._terminal = True

        observation = self._render_rgb()
        return observation, float(reward), bool(terminated), truncated, self._make_info()


def make_corridortv_env(**kwargs) -> gym.Env:
    """Construct the registered CorridorTV environment.

    Input: Keyword arguments forwarded to ``CorridorTVEnv``.
    Output: A ``TimeLimit``-wrapped ``CorridorTVEnv`` with a 1000-step budget.
    Mathematical meaning: Factory used by ``gymnasium.make`` to produce the
        time-limited CorridorTV environment.
    """
    max_episode_steps = kwargs.pop("max_episode_steps", CORRIDORTV_MAX_EPISODE_STEPS)
    return wrappers.TimeLimit(
        CorridorTVEnv(max_episode_steps=max_episode_steps),
        max_episode_steps=max_episode_steps,
    )


# ---------------------------------------------------------------------------
# Registration. ``gymnasium.make("CorridorTV-v0")`` resolves this spec. The env
# internally truncates at ``max_episode_steps``; ``make_corridortv_env`` also
# wraps it in ``gymnasium.wrappers.TimeLimit`` for explicit use.
# ---------------------------------------------------------------------------
gym.register(
    id="CorridorTV-v0",
    entry_point="environments.corridortv:CorridorTVEnv",
    max_episode_steps=CORRIDORTV_MAX_EPISODE_STEPS,
    order_enforce=False,
)
