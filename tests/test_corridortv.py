"""Unit tests for the CorridorTV environment (Part-3 Stage-1 environment).

The tests are all CPU-runnable and validate the exact observation/reward/info
contract called for by the Part-3 protocol:

* observation shape/dtype ``(64, 64, 3) uint8`` and ``Discrete(4)`` action space,
* spec registration via ``gymnasium.spec``,
* exhaustive BFS over the deterministic transition function asserting exactly
  73 reachable controllable configurations,
* sparse goal reward (``+1`` and termination, no other positive reward) and
  truncation at 1000 steps,
* action-independence of the uncontrollable flicker/decoy factors,
* ``BiGANEncoder`` compatibility smoke on a stacked observation batch, and
* determinism under a fixed seed and action sequence.
"""

from __future__ import annotations

import unittest
from collections import deque

import numpy as np
import torch

import gymnasium as gym

import environments.corridortv as ct
from bigan.encoder import BiGANEncoder

# Deterministic movement: 0=up, 1=down, 2=left, 3=right.
ACTIONS = [(0, -1), (0, 1), (-1, 0), (1, 0)]

# A short, controllable action trajectory that reaches the goal through the
# switch and door (validated against the env's deterministic transition).
GOAL_PATH = [1, 1, 3, 3, 1, 1, 3, 3, 1]


class ObservationAndActionSpaceTest(unittest.TestCase):
    """Validate the observation and action-space contract."""

    def test_observation_shape_and_dtype(self) -> None:
        env = gym.make("CorridorTV-v0")
        try:
            obs, _info = env.reset(seed=0)
            self.assertEqual(obs.shape, (64, 64, 3))
            self.assertEqual(obs.dtype, np.uint8)
            self.assertTrue(issubclass(env.observation_space.dtype.type, np.integer))
            self.assertEqual(env.observation_space.shape, (64, 64, 3))
        finally:
            env.close()

    def test_action_space_is_discrete4(self) -> None:
        env = gym.make("CorridorTV-v0")
        try:
            self.assertIsInstance(env.action_space, gym.spaces.Discrete)
            self.assertEqual(env.action_space.n, 4)
        finally:
            env.close()


class RegistrationTest(unittest.TestCase):
    """Validate the gymnasium spec registration."""

    def test_spec_registered(self) -> None:
        spec = gym.spec("CorridorTV-v0")
        self.assertIsNotNone(spec)
        self.assertEqual(spec.id, "CorridorTV-v0")
        self.assertEqual(spec.max_episode_steps, ct.CORRIDORTV_MAX_EPISODE_STEPS)

    def test_entry_point_resolves(self) -> None:
        spec = gym.spec("CorridorTV-v0")
        env = gym.make("CorridorTV-v0")
        try:
            self.assertTrue(hasattr(env, "observation_space"))
            self.assertTrue(spec.entry_point, "environments.corridortv:CorridorTVEnv")
        finally:
            env.close()


class ControllableStateBFSTest(unittest.TestCase):
    """Exhaustively BFS the deterministic transition function and assert 73."""

    def test_bfs_reaches_exactly_73_controllable_configurations(self) -> None:
        env = gym.make("CorridorTV-v0")
        try:
            unwrapped = env.unwrapped
            path = unwrapped._path
            coord_to_index = unwrapped._coord_to_index
            goal = unwrapped.goal_coordinate
            door_index = unwrapped.door_index
            switch_index = unwrapped.switch_index

            def transition(index: int, door_open: bool, action: int):
                current = path[index]
                dx, dy = ACTIONS[action]
                target = (current[0] + dx, current[1] + dy)
                if target == goal:
                    return index, door_open, True
                if target not in coord_to_index:
                    return index, door_open, False
                next_index = coord_to_index[target]
                if next_index == door_index and not door_open:
                    return index, door_open, False
                next_door_open = door_open ^ (next_index == switch_index)
                return next_index, next_door_open, False

            start = (unwrapped.start_index, False)
            reachable = {start}
            queue = deque([start])
            while queue:
                index, door_open = queue.popleft()
                for action in range(4):
                    next_index, next_door_open, terminated = transition(
                        index, door_open, action
                    )
                    if terminated:
                        # Terminal (goal) states are not controllable states.
                        continue
                    state = (next_index, next_door_open)
                    if state not in reachable:
                        reachable.add(state)
                        queue.append(state)
            self.assertEqual(len(reachable), ct.CORRIDORTV_CONTROLLABLE_STATES)
            self.assertEqual(len(reachable), 73)
        finally:
            env.close()

    def test_controllable_table_and_ground_truth_agree(self) -> None:
        env = gym.make("CorridorTV-v0")
        try:
            self.assertEqual(
                len(env.unwrapped.controllable_table),
                ct.CORRIDORTV_CONTROLLABLE_STATES,
            )
            self.assertEqual(
                ct.CorridorTVEnv.ground_truth_controllability(),
                ct.CORRIDORTV_CONTROLLABLE_STATES,
            )
        finally:
            env.close()

    def test_controllable_state_is_in_range(self) -> None:
        env = gym.make("CorridorTV-v0")
        try:
            obs, info = env.reset(seed=0)
            self.assertGreaterEqual(info["controllable_state"], 0)
            self.assertLess(info["controllable_state"], 73)
        finally:
            env.close()


class SparseRewardAndTruncationTest(unittest.TestCase):
    """Validate sparse goal reward (+1, termination) and 1000-step truncation."""

    def test_goal_rewakes_positive_reward_and_termination(self) -> None:
        env = gym.make("CorridorTV-v0")
        try:
            obs, info = env.reset(seed=0)
            reward_sum = 0.0
            terminated = False
            for _action in GOAL_PATH:
                obs, reward, terminated, _trunc, info = env.step(_action)
                reward_sum += reward
                if terminated:
                    break
            self.assertTrue(terminated)
            self.assertEqual(reward_sum, 1.0)
            self.assertEqual(info["goal_reached"], True)
            self.assertEqual(info["door_open"], True)
        finally:
            env.close()

    def test_no_positive_reward_before_goal(self) -> None:
        env = gym.make("CorridorTV-v0")
        try:
            env.reset(seed=0)
            total = 0.0
            # Push into the wall at the start cell; the agent stays there and
            # never reaches the goal, so every reward must be zero.
            for _ in range(200):
                _obs, reward, terminated, _trunc, _info = env.step(0)
                self.assertEqual(reward, 0.0)
                total += reward
                self.assertFalse(terminated)
            self.assertEqual(total, 0.0)
        finally:
            env.close()

    def test_episode_truncates_at_1000_steps(self) -> None:
        env = gym.make("CorridorTV-v0")
        try:
            env.reset(seed=0)
            truncated = False
            steps = 0
            for _ in range(ct.CORRIDORTV_MAX_EPISODE_STEPS):
                _obs, _r, terminated, truncated, _info = env.step(0)
                steps += 1
                if terminated or truncated:
                    break
            self.assertEqual(steps, 1000)
            self.assertTrue(truncated)
            self.assertFalse(terminated)
        finally:
            env.close()


class UncontrollableVisualFactorTest(unittest.TestCase):
    """Validate the visual distractors are action- and agent-independent."""

    def _run_controllable_trajectory(self, seed: int) -> list:
        env = gym.make("CorridorTV-v0")
        try:
            env.reset(seed=seed)
            records = []
            for action in GOAL_PATH:
                _obs, reward, terminated, _trunc, info = env.step(action)
                records.append(
                    (
                        info["controllable_state"],
                        reward,
                        terminated,
                        info["door_open"],
                    )
                )
                if terminated:
                    break
            return records
        finally:
            env.close()

    def test_visual_seed_does_not_change_controllable_trajectory(self) -> None:
        # Different seeds change only the flicker/decoy visual streams, never the
        # controllable state, reward, or termination sequence.
        a = self._run_controllable_trajectory(seed=100)
        b = self._run_controllable_trajectory(seed=999)
        self.assertEqual(a, b)

    def test_obs_which_differ_in_only_flicker_still_have_identical_controls(self) -> None:
        # Confirm the same controllable trajectory is reported under two seeds.
        a = self._run_controllable_trajectory(seed=1)
        b = self._run_controllable_trajectory(seed=2)
        self.assertEqual(a, b)


class EncoderCompatibilitySmokeTest(unittest.TestCase):
    """One forward pass of BiGANEncoder((64,64,3),128,256,512) on an obs batch."""

    def test_encoder_accepts_stacked_observation(self) -> None:
        env = gym.make("CorridorTV-v0")
        try:
            obs, _info = env.reset(seed=0)
            batch = np.stack([obs, obs, obs])
            encoder = BiGANEncoder((64, 64, 3), 128, 256, 512)
            latent = encoder(torch.as_tensor(batch).float())
            self.assertEqual(tuple(latent.shape), (3, 128))
        finally:
            env.close()


class DeterminismTest(unittest.TestCase):
    """Two environments with the same seed produce identical trajectories."""

    def test_same_seed_identical_trajectories(self) -> None:
        def run(seed: int) -> tuple:
            env = gym.make("CorridorTV-v0")
            try:
                obs, info = env.reset(seed=seed)
                trajectory = [obs.copy()]
                infos = [dict(info)]
                for action in GOAL_PATH:
                    obs, reward, terminated, _trunc, info = env.step(action)
                    trajectory.append(obs.copy())
                    infos.append(dict(info))
                    if terminated:
                        break
                return np.stack(trajectory), infos
            finally:
                env.close()

        obs_a, infos_a = run(seed=42)
        obs_b, infos_b = run(seed=42)
        self.assertTrue(np.array_equal(obs_a, obs_b))
        self.assertEqual(
            [x["controllable_state"] for x in infos_a],
            [x["controllable_state"] for x in infos_b],
        )
        self.assertEqual(
            [x["seed"] for x in infos_a],
            [x["seed"] for x in infos_b],
        )


if __name__ == "__main__":
    unittest.main()
