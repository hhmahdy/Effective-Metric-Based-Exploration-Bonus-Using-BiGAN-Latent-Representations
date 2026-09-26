"""Tests for the default environment of the Master's experiment driver.

``scripts/run_master_experiments.sh`` runs ``FetchPickAndPlace-v4`` unless
``ENV_NAME`` says otherwise. The environment is a goal-conditioned,
sparse-reward manipulation task: the observation is the usual dictionary
(``observation`` + ``achieved_goal`` + ``desired_goal``), the action space is
``Box(-1, 1, (4,))``, and the reward is ``-1`` per step with ``0`` on success,
so a run that reaches the goal at all is the signal an exploration bonus is
meant to produce.

These tests exist for two reasons:

* the driver's default must not drift silently - it is read out of the script
  itself, not restated;
* the environment must keep working end to end through the pipeline
  (``gymnasium.make`` -> ``SingleEnvironmentAdapter`` -> ``main.build_config``),
  including the continuous-action path, because the default environment is the
  one a reader of the README will run first. The MuJoCo pin in
  ``requirements-master.txt`` is part of that contract: gymnasium-robotics 1.4.2
  cannot construct the Fetch tasks against MuJoCo 3.14.
"""

from __future__ import annotations

import re
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from environments.adapter import SingleEnvironmentAdapter, make_gymnasium_environment

REPOSITORY_ROOT = Path(__file__).resolve().parent.parent
DRIVER = REPOSITORY_ROOT / "scripts" / "run_master_experiments.sh"
DEFAULT_ENVIRONMENT = "FetchPickAndPlace-v4"


def _gymnasium_robotics_available() -> bool:
    """Report whether the default environment's stack is importable.

    Input: None.
    Output: ``True`` when ``gymnasium_robotics`` and MuJoCo are importable.
    Mathematical meaning: None; it decides whether the MuJoCo-backed tests run.
    """
    try:
        import gymnasium_robotics  # noqa: F401
    except ImportError:
        return False
    try:
        import mujoco  # noqa: F401
    except ImportError:
        return False
    return True


def _default_environment_name() -> str:
    """Read the ``ENV_NAME`` default out of the driver script.

    Input: None.
    Output: The literal default in ``ENV_NAME="${ENV_NAME:-...}"``.
    Mathematical meaning: None; it documents the shipped default.

    Raises:
        AssertionError: If the script does not contain the assignment.
    """
    match = re.search(
        r'^ENV_NAME="\$\{ENV_NAME:-(?P<name>[^}"]+)\}"',
        DRIVER.read_text(),
        flags=re.MULTILINE,
    )
    assert match is not None, f"no ENV_NAME default found in {DRIVER}"
    return match.group("name")


class DriverDefaultTest(unittest.TestCase):
    """The driver's default environment, as shipped."""

    def test_driver_defaults_to_fetch_pick_and_place(self) -> None:
        """``ENV_NAME`` defaults to ``FetchPickAndPlace-v4``.

        Input: The driver script.
        Output: Assertion on the parsed default and on the documented name.
        Mathematical meaning: None; it fixes which MDP the thesis experiment
            samples by default.
        """
        self.assertEqual(_default_environment_name(), DEFAULT_ENVIRONMENT)
        script = DRIVER.read_text()
        self.assertIn(f"[{DEFAULT_ENVIRONMENT}]", script, "the usage text must state the default")

    def test_python_entry_point_defaults_to_the_same_environment(self) -> None:
        """``main.py`` without ``--env`` trains on ``FetchPickAndPlace-v4``.

        Input: ``main.parse_args`` driven with no environment argument.
        Output: Assertions on the parsed id and on the module constant.
        Mathematical meaning: None; it fixes the environment ``config.json``
            and ``run_info.json`` record for a default run.
        """
        import main as main_module

        with mock.patch.object(sys, "argv", ["main.py"]):
            arguments = main_module.parse_args()
        self.assertEqual(arguments.environment_id, DEFAULT_ENVIRONMENT)
        self.assertEqual(main_module.DEFAULT_ENVIRONMENT_ID, DEFAULT_ENVIRONMENT)
        # The driver and the python entry point must not drift apart.
        self.assertEqual(_default_environment_name(), main_module.DEFAULT_ENVIRONMENT_ID)

    def test_driver_documents_that_the_default_is_configurable(self) -> None:
        """The driver keeps the environment overridable by name.

        Input: The driver script.
        Output: Assertion that the override and one alternative are documented.
        Mathematical meaning: None; it keeps the other environments reachable.
        """
        script = DRIVER.read_text()
        self.assertIn("ENV_NAME=NoisyTVMaze-v0", script)
        self.assertIn("ENV_NAME=ALE/MontezumaRevenge-v5", script)

    def test_default_environment_needs_the_robotics_stack(self) -> None:
        """The requirements manifest pins the stack the default environment needs.

        Input: ``requirements-master.txt``.
        Output: Assertions that gymnasium-robotics and MuJoCo are pinned.
        Mathematical meaning: None; the default environment is not installable
            without them.
        """
        requirements = (REPOSITORY_ROOT / "requirements-master.txt").read_text()
        self.assertIn("gymnasium-robotics==", requirements)
        self.assertIn("mujoco==", requirements)


@unittest.skipUnless(
    _gymnasium_robotics_available(),
    "gymnasium-robotics or MuJoCo is not installed",
)
class FetchPickAndPlaceTest(unittest.TestCase):
    """The default environment must drive the pipeline unchanged."""

    def test_environment_has_the_expected_spaces(self) -> None:
        """The task exposes the dictionary observation and continuous actions.

        Input: ``gymnasium.make`` on the default ID.
        Output: Assertions on the spaces and on one transition.
        Mathematical meaning: Defines the state domain and the action support
            the policy must cover.
        """
        import gymnasium as gym
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        environment = gym.make(DEFAULT_ENVIRONMENT, max_episode_steps=50)
        try:
            self.assertIsInstance(environment.observation_space, gym.spaces.Dict)
            self.assertEqual(
                sorted(environment.observation_space.spaces),
                ["achieved_goal", "desired_goal", "observation"],
            )
            self.assertEqual(environment.action_space.shape, (4,))
            self.assertTrue(environment.action_space.is_bounded())
            observation, _ = environment.reset(seed=0)
            result = environment.step(environment.action_space.sample())
            self.assertEqual(len(result), 5, "gymnasium's five-tuple contract")
            self.assertEqual(result[1], -1.0, "every step before success costs -1")
        finally:
            environment.close()

    def test_pipeline_adapter_flattens_and_steps(self) -> None:
        """The pipeline adapter flattens the dictionary observation.

        Input: The default ID through ``make_gymnasium_environment``.
        Output: Assertions on the flattened shape and on a tensor step.
        Mathematical meaning: The policy, critic, and BiGAN see one fixed
            vector: 25 + 3 + 3 = 31 components.
        """
        environment = make_gymnasium_environment(DEFAULT_ENVIRONMENT)
        try:
            self.assertEqual(environment.observation_shape, (31,))
            observation, _ = environment.reset()
            self.assertEqual(tuple(observation.shape), (31,))
            step = environment.step(
                np.zeros(4, dtype=np.float32)
            )
            self.assertEqual(tuple(step.observation.shape), (31,))
            self.assertIsInstance(step.reward, float)
        finally:
            environment.close()

    def test_build_config_accepts_the_default_environment(self) -> None:
        """``main.build_config`` derives a continuous-action configuration.

        Input: ``--env FetchPickAndPlace-v4`` arguments and the wrapped
            environment.
        Output: Assertions on the observation shape, the action dimension, and
            on continuous-action handling.
        Mathematical meaning: The PPO/BiGAN function domains follow this MDP
            instead of the discrete Noisy-TV ones.
        """
        import main as main_module

        environment = make_gymnasium_environment(DEFAULT_ENVIRONMENT)
        try:
            # ``main.parse_args`` reads sys.argv directly, so drive the real
            # entry point instead of restating its defaults.
            with mock.patch.object(
                sys,
                "argv",
                [
                    "main.py",
                    "--env",
                    DEFAULT_ENVIRONMENT,
                    "--method",
                    "transition",
                    "--num-parallel-envs",
                    "2",
                ],
            ):
                arguments = main_module.parse_args()
            configuration = main_module.build_config(arguments, environment)
        finally:
            environment.close()
        self.assertEqual(configuration.environment.observation_shape, (31,))
        self.assertEqual(configuration.environment.action_dim, 4)
        self.assertFalse(configuration.environment.discrete_actions)

    def test_recorded_configuration_describes_the_environment(self) -> None:
        """``config.json`` records the environment's real limits and inputs.

        Input: ``build_config`` for the default environment.
        Output: Assertions on the recorded id, horizon, and preprocessing.
        Mathematical meaning: The configuration file is the record of the
            experiment, so the episode length ``T`` and the state
            representation it reports must be the ones the run used.

        The dataclass defaults describe the 84x84x4 Atari/Noisy-TV setting; a
        50-step vector-observation task must not inherit them.
        """
        import main as main_module

        environment = make_gymnasium_environment(DEFAULT_ENVIRONMENT)
        try:
            with mock.patch.object(sys, "argv", ["main.py", "--method", "transition"]):
                arguments = main_module.parse_args()
            configuration = main_module.build_config(arguments, environment)
        finally:
            environment.close()
        recorded = configuration.environment
        self.assertEqual(recorded.environment_id, DEFAULT_ENVIRONMENT)
        self.assertEqual(recorded.max_episode_steps, 50)
        self.assertEqual(recorded.frame_stack, 1)
        self.assertFalse(recorded.normalize_pixels)

    def test_parallel_environments_are_independent(self) -> None:
        """Several Fetch environments can be built for batched rollouts.

        Input: ``main.make_training_environment`` with two parallel
            environments.
        Output: Assertions on the batched observation and action shapes.
        Mathematical meaning: The product MDP of two independent task
            instances is what the driver samples each rollout.
        """
        import main as main_module

        vector_environment = main_module.make_training_environment(
            DEFAULT_ENVIRONMENT, num_parallel_envs=2, seed=0
        )
        try:
            observations, _ = vector_environment.reset()
            self.assertEqual(tuple(observations.shape)[0], 2)
            self.assertEqual(tuple(observations.shape)[1:], (31,))
            transitions = vector_environment.step(
                np.zeros((2, 4), dtype=np.float32)
            )
            self.assertEqual(len(transitions.rewards), 2)
            self.assertTrue(all(np.isfinite(transitions.rewards)))
        finally:
            vector_environment.close()


class SingleEnvironmentAdapterContractTest(unittest.TestCase):
    """The adapter contract the Fetch task relies on, checked without MuJoCo."""

    def test_dictionary_observations_are_flattened_in_key_order(self) -> None:
        """Dictionary observations are concatenated in the documented order.

        Input: A stub environment with a three-key ``Dict`` observation space.
        Output: Assertions on the flattened values and shape.
        Mathematical meaning: Fixes the component order of the state vector
            ``s_t`` presented to the policy, critic, and BiGAN.
        """
        import gymnasium as gym

        class _Stub(gym.Env):
            """Minimal goal-conditioned environment with a dictionary space."""

            observation_space = gym.spaces.Dict(
                {
                    "observation": gym.spaces.Box(-np.inf, np.inf, (2,)),
                    "achieved_goal": gym.spaces.Box(-np.inf, np.inf, (1,)),
                    "desired_goal": gym.spaces.Box(-np.inf, np.inf, (1,)),
                }
            )
            action_space = gym.spaces.Box(-1.0, 1.0, (4,), dtype=np.float32)

            def reset(self, *, seed=None, options=None):
                del seed, options
                return (
                    {
                        "observation": np.array([1.0, 2.0], dtype=np.float32),
                        "achieved_goal": np.array([3.0], dtype=np.float32),
                        "desired_goal": np.array([4.0], dtype=np.float32),
                    },
                    {},
                )

            def step(self, action):
                observation, _ = self.reset()
                return observation, -1.0, False, False, {}

        environment = SingleEnvironmentAdapter(_Stub())
        try:
            self.assertEqual(environment.observation_shape, (4,))
            observation, _ = environment.reset()
            self.assertEqual(observation.shape, (4,))
            self.assertAlmostEqual(float(observation[0]), 1.0)
            self.assertAlmostEqual(float(observation[1]), 2.0)
            self.assertAlmostEqual(float(observation[2]), 3.0)
            self.assertAlmostEqual(float(observation[3]), 4.0)
        finally:
            environment.close()


if __name__ == "__main__":
    unittest.main()

class RecordedEnvironmentMetadataTest(unittest.TestCase):
    """The record must follow the environment, not the image defaults."""

    def _recorded_environment(self, environment_id: str):
        """Build the configuration a run of ``environment_id`` would record.

        Input: Registered Gymnasium id.
        Output: The ``EnvironmentConfig`` written to ``config.json``.
        Mathematical meaning: None; it exercises the metadata derivation.
        """
        import main as main_module

        environment = make_gymnasium_environment(environment_id)
        try:
            with mock.patch.object(
                sys, "argv", ["main.py", "--env", environment_id, "--method", "state"]
            ):
                arguments = main_module.parse_args()
            return main_module.build_config(arguments, environment).environment
        finally:
            environment.close()

    def test_image_environment_keeps_pixel_preprocessing(self) -> None:
        """A ``uint8`` image environment is recorded as pixels, not vectors.

        Input: ``NoisyTVMaze-v0``, whose observations are 84x84x3 ``uint8``.
        Output: Assertions on the recorded id, horizon, and preprocessing.
        Mathematical meaning: Confirms the derivation distinguishes pixel
            states from vector states instead of always answering one way.
        """
        recorded = self._recorded_environment("NoisyTVMaze-v0")
        self.assertEqual(recorded.environment_id, "NoisyTVMaze-v0")
        self.assertEqual(recorded.observation_shape, (84, 84, 3))
        self.assertEqual(recorded.max_episode_steps, 1000)
        self.assertEqual(recorded.frame_stack, 4)
        self.assertTrue(recorded.normalize_pixels)

    def test_vector_environment_records_its_own_horizon(self) -> None:
        """A vector environment records its registration horizon.

        Input: ``CartPole-v1``, registered with a 500-step limit.
        Output: Assertions on the recorded horizon and preprocessing.
        Mathematical meaning: The horizon is read from the environment's
            ``spec``, not assumed to be the image default.
        """
        recorded = self._recorded_environment("CartPole-v1")
        self.assertEqual(recorded.environment_id, "CartPole-v1")
        self.assertEqual(recorded.observation_shape, (4,))
        self.assertEqual(recorded.max_episode_steps, 500)
        self.assertEqual(recorded.frame_stack, 1)
        self.assertFalse(recorded.normalize_pixels)
