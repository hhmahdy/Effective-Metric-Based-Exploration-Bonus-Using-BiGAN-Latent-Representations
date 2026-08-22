"""Vectorized Adventurer training pipeline.

The trainer implements separate extrinsic/intrinsic reward streams, two-stream
GAE, PPO with two critics, BiGAN novelty, optional resettable episodic memory,
and N parallel environments. With the default Atari configuration it collects
96*128=12,288 transitions per PPO rollout.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch
from torch import Tensor

from agent.actor import ActorNetwork
from agent.advantage import (
    TwoStreamAdvantageEstimate,
    compute_two_stream_returns_and_advantages,
)
from agent.critic import DualCriticNetwork
from agent.ppo import PPOAgent, PPOUpdateMetrics
from agent.rollout_buffer import RolloutBuffer
from bigan.discriminator import BiGANDiscriminator
from bigan.encoder import BiGANEncoder
from bigan.generator import BiGANGenerator
from bigan.trainer import BiGANTrainer, BiGANUpdateMetrics
from config import ExperimentConfig
from environments.vector_adapter import VectorEnvironmentAdapter
from exploration.episodic_memory import EpisodicMemory
from exploration.intrinsic_reward import IntrinsicRewardPipeline
from exploration.novelty import NoveltyEstimator
from exploration.transition_novelty import (
    LatentForwardModel,
    TransitionNoveltyEstimator,
    TransitionNoveltyTrainer,
    TransitionUpdateMetrics,
)
from utils.logger import ExperimentLogger
from utils.replay import ReplayBuffer
from utils.transition_replay import TransitionReplayBuffer
from utils.seed import create_torch_generator, derive_seed, seed_everything
from utils.visualization import TrainingPlotter


@dataclass(frozen=True)
class TrainingIterationMetrics:
    """Diagnostics from one vectorized rollout and update cycle."""

    update: int
    environment_steps: int
    extrinsic_reward: float
    intrinsic_reward: float
    total_reward: float
    pixel_novelty: float
    feature_novelty: float
    normalized_novelty: float
    ppo: PPOUpdateMetrics
    bigan: BiGANUpdateMetrics
    transition: Optional[TransitionUpdateMetrics] = None


class AdventurerTrainer:
    """Train Adventurer with N parallel environments."""

    def __init__(
        self,
        environment: VectorEnvironmentAdapter,
        config: ExperimentConfig,
        logger: Optional[ExperimentLogger] = None,
    ) -> None:
        """Construct vectorized networks, buffers, and exploration modules.

        Input: Vector environment, complete configuration, and optional logger.
        Output: Initialized trainer with ``N`` parallel rollout streams.
        Mathematical meaning: Instantiates the product-MDP policy/value,
            BiGAN, and two independent reward-stream optimization processes.
        """
        self.environment = environment
        self.config = config
        self.device = self._resolve_device(config.training.device.value)
        self.num_envs = environment.num_envs
        if self.num_envs != config.environment.num_parallel_envs:
            raise ValueError("environment count does not match configuration")
        seed_everything(config.seed)
        self.environment.seed(config.seed.seed)
        self.logger = logger
        self.resettable = config.training.resettable
        self.episodic_memory = EpisodicMemory(config.training.episodic_memory_size)
        self.memory_generator = create_torch_generator(
            derive_seed(config.seed.seed, 0xE91), device="cpu"
        )
        if self.resettable and not self.environment.supports_state_restore():
            raise RuntimeError(
                "resettable=True requires simulator clone/restore support in every environment"
            )

        plot_directory = config.training.plot_directory
        if plot_directory is None:
            plot_directory = f"{config.training.output_directory}/figures"
        self.plotter = (
            TrainingPlotter(
                output_directory=plot_directory,
                smoothing_window=config.training.score_smoothing_window,
                interval_updates=config.training.plot_interval_updates,
                figure_width=config.training.figure_width,
                figure_height=config.training.figure_height,
                dpi=config.training.figure_dpi,
            )
            if config.training.save_plots
            else None
        )
        observation_shape = config.environment.observation_shape
        action_dim = config.environment.action_dim
        self.actor = ActorNetwork(
            observation_shape, action_dim, config.environment.discrete_actions
        ).to(self.device)
        self.critic = DualCriticNetwork(observation_shape).to(self.device)
        total_updates = config.training.total_environment_steps // (
            config.ppo.rollout_steps * self.num_envs
        )
        if total_updates <= 0:
            raise ValueError("training budget must contain at least one complete vector rollout")
        self.ppo = PPOAgent(
            self.actor,
            self.critic,
            config.ppo,
            total_updates,
            beta=config.ppo.intrinsic_advantage_coefficient,
            device=self.device,
        )
        self.encoder = BiGANEncoder(
            observation_shape,
            config.bigan.latent_dim,
            config.bigan.feature_dim,
            config.bigan.hidden_dim,
        )
        self.generator = BiGANGenerator(
            observation_shape,
            config.bigan.latent_dim,
            config.bigan.feature_dim,
            config.bigan.hidden_dim,
        )
        self.discriminator = BiGANDiscriminator(
            observation_shape,
            config.bigan.latent_dim,
            config.bigan.feature_dim,
            config.bigan.hidden_dim,
        )
        self.bigan = BiGANTrainer(
            self.encoder,
            self.generator,
            self.discriminator,
            config.bigan,
            observation_shape,
            self.device,
        )
        self.novelty = None
        self.transition_novelty = None
        self.transition_trainer = None
        self.transition_replay_buffer = None
        if config.novelty.novelty_type == "state":
            self.novelty = NoveltyEstimator(
                self.encoder,
                self.generator,
                self.discriminator,
                config.novelty.alpha,
                config.novelty.normalization_epsilon,
                device=self.device,
            )
        else:
            forward_model = LatentForwardModel(
                config.bigan.latent_dim,
                action_dim,
                config.novelty.transition_hidden_dim,
                config.environment.discrete_actions,
            )
            self.transition_novelty = TransitionNoveltyEstimator(
                self.encoder,
                self.generator,
                self.discriminator,
                forward_model,
                config.novelty.transition_alpha,
                config.novelty.normalization_epsilon,
                device=self.device,
            )
            self.transition_trainer = TransitionNoveltyTrainer(
                self.transition_novelty,
                config.novelty.transition_learning_rate,
                config.novelty.transition_update_epochs,
                config.novelty.transition_max_grad_norm,
            )
            self.transition_replay_buffer = TransitionReplayBuffer(
                max(config.ppo.rollout_steps * self.num_envs, config.bigan.batch_size),
                observation_shape,
                () if config.environment.discrete_actions else (action_dim,),
                self.device,
                torch.float32,
                torch.long if config.environment.discrete_actions else torch.float32,
            )
        self.reward_pipeline = IntrinsicRewardPipeline(config.novelty.clip_intrinsic_reward)
        self.rollout_buffer = RolloutBuffer(
            config.ppo.rollout_steps,
            self.num_envs,
            observation_shape,
            () if config.environment.discrete_actions else (action_dim,),
            self.device,
            torch.float32,
            torch.long if config.environment.discrete_actions else torch.float32,
        )
        self.replay_buffer = ReplayBuffer(
            max(config.ppo.rollout_steps * self.num_envs, config.bigan.batch_size),
            observation_shape,
            self.device,
            torch.float32,
        )
        self._observation: Optional[Tensor] = None
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float64)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.long)
        self.environment_steps = 0

    @staticmethod
    def _resolve_device(device_name: str) -> torch.device:
        """Resolve automatic, CPU, or CUDA device selection.

        Input: Device policy string.
        Output: Usable torch device.
        Mathematical meaning: Selects the substrate for batched optimization.
        """
        if device_name == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        return torch.device(device_name)

    def _reset_if_needed(self) -> Tensor:
        """Reset all environments when the trainer has no active batch.

        Input: Trainer state.
        Output: Batched observation tensor ``[N,...]``.
        Mathematical meaning: Establishes the initial product-MDP state batch.
        """
        if self._observation is None:
            observations, _ = self.environment.reset(
                [derive_seed(self.config.seed.seed, index) for index in range(self.num_envs)]
            )
            self._observation = observations.to(self.device)
        return self._observation

    def _reset_done(self, observations: Tensor, done: Tensor) -> Tensor:
        """Reset completed environment slots, optionally from episodic memory.

        Input: Terminal next-observation batch and boolean done mask ``[N]``.
        Output: Next current-observation batch with completed slots reset.
        Mathematical meaning: Starts new trajectories in the product MDP while
            preserving active trajectories.
        """
        current = observations.clone()
        for index in torch.nonzero(done, as_tuple=False).flatten().tolist():
            memory_entry = (
                self.episodic_memory.sample_previous(self.memory_generator)
                if self.resettable
                else None
            )
            if memory_entry is not None:
                if memory_entry.environment_snapshot is None:
                    raise RuntimeError("episodic memory entry lacks simulator snapshot")
                current[index] = self.environment.restore_state(
                    index, memory_entry.environment_snapshot
                )
            else:
                current[index] = self.environment.reset_one(
                    index,
                    derive_seed(self.config.seed.seed, self.environment_steps + index),
                )
            self.episode_returns[index] = 0.0
            self.episode_lengths[index] = 0
        return current

    def _record_completed_episodes(self, done: Tensor) -> None:
        """Record scores and lengths for all episodes ending in one batch step.

        Input: Boolean done mask ``[N]``.
        Output: No value; completed scores are sent to plotter and logger.
        Mathematical meaning: Records empirical episodic extrinsic returns.
        """
        for index in torch.nonzero(done, as_tuple=False).flatten().tolist():
            score = float(self.episode_returns[index].item())
            length = int(self.episode_lengths[index].item())
            timestep = self.environment_steps + index
            if self.plotter is not None:
                self.plotter.record_episode(timestep, score, length)
            if self.logger is not None:
                self.logger.log(
                    timestep,
                    {"episode/game_score": score, "episode/length": length},
                )

    def _collect_rollout(self) -> Dict[str, float]:
        """Collect T batched transitions and preserve both reward streams.

        Input: Current trainer state and configured rollout horizon T.
        Output: Mean reward and novelty diagnostics across ``T*N`` samples.
        Mathematical meaning: Stores separate ``r^e``, ``r^i``, ``V^e``, and
            ``V^i`` trajectories for two-stream GAE.
        """
        self.rollout_buffer.reset()
        if self.resettable:
            self.episodic_memory.begin_epoch()
        sums = {
            "extrinsic": 0.0,
            "intrinsic": 0.0,
            "total": 0.0,
            "pixel": 0.0,
            "feature": 0.0,
            "normalized": 0.0,
        }
        for _ in range(self.config.ppo.rollout_steps):
            observations = self._reset_if_needed()
            actions, log_probabilities, extrinsic_values, intrinsic_values = self.ppo.select_action(observations)
            transition = self.environment.step(actions)
            next_observations = transition.observations.to(self.device)
            extrinsic_rewards = transition.rewards.to(self.device)
            if self.config.novelty.novelty_type == "state":
                assert self.novelty is not None
                novelty = self.novelty(
                    next_observations,
                    update_statistics=True,
                    extrinsic_reward=extrinsic_rewards,
                )
            else:
                assert self.transition_novelty is not None
                novelty = self.transition_novelty(
                    observations,
                    actions,
                    next_observations,
                    extrinsic_rewards,
                )
                assert self.transition_replay_buffer is not None
                self.transition_replay_buffer.add_batch(
                    observations.detach(), actions.detach(), next_observations.detach()
                )
            reward_result = self.reward_pipeline(extrinsic_rewards, novelty)
            if self.resettable:
                snapshots = self.environment.clone_states()
                for index in range(self.num_envs):
                    self.episodic_memory.add(
                        next_observations[index],
                        float(novelty.combined_score[index].item()),
                        snapshots[index],
                    )
            self.rollout_buffer.add(
                observations,
                actions,
                reward_result.extrinsic_reward,
                reward_result.intrinsic_reward,
                transition.terminated,
                transition.truncated,
                extrinsic_values,
                intrinsic_values,
                log_probabilities,
                next_observations,
            )
            self.replay_buffer.add_batch(next_observations.detach())
            done = transition.done
            self.episode_returns += extrinsic_rewards.detach().double().cpu()
            self.episode_lengths += 1
            self._record_completed_episodes(done.cpu())
            self.environment_steps += self.num_envs
            self._observation = self._reset_done(next_observations, done)
            sums["extrinsic"] += float(reward_result.extrinsic_reward.detach().sum().cpu())
            sums["intrinsic"] += float(reward_result.intrinsic_reward.detach().sum().cpu())
            sums["total"] += float((reward_result.extrinsic_reward + reward_result.intrinsic_reward).detach().sum().cpu())
            sums["pixel"] += float(novelty.pixel_error.detach().sum().cpu())
            sums["feature"] += float(novelty.feature_error.detach().sum().cpu())
            sums["normalized"] += float(novelty.normalized_score.detach().sum().cpu())
        self.rollout_buffer.set_final_observation(self._reset_if_needed())
        denominator = float(self.config.ppo.rollout_steps * self.num_envs)
        return {key: value / denominator for key, value in sums.items()}

    def train(self) -> List[TrainingIterationMetrics]:
        """Run vectorized collection, two-stream GAE, PPO, and BiGAN updates.

        Input: No arguments; uses immutable experiment configuration.
        Output: Per-update diagnostics.
        Mathematical meaning: Computes independent ``A^e`` and ``A^i`` over
            ``T*N`` samples, then PPO uses ``A^e+beta*A^i``.
        """
        results: List[TrainingIterationMetrics] = []
        samples_per_update = self.config.ppo.rollout_steps * self.num_envs
        total_updates = self.config.training.total_environment_steps // samples_per_update
        for update_index in range(total_updates):
            summary = self._collect_rollout()
            tensors = self.rollout_buffer.tensors()
            extrinsic_last_value, intrinsic_last_value = self.critic(
                self.rollout_buffer.observations[-1]
            )
            estimate: TwoStreamAdvantageEstimate = compute_two_stream_returns_and_advantages(
                tensors["extrinsic_rewards"],
                tensors["intrinsic_rewards"],
                tensors["extrinsic_values"],
                tensors["intrinsic_values"],
                tensors["terminated"],
                tensors["truncated"],
                extrinsic_last_value.detach(),
                intrinsic_last_value.detach(),
                self.config.ppo.gamma,
                self.config.ppo.gae_lambda,
                self.config.ppo.intrinsic_gamma,
                self.config.ppo.intrinsic_gae_lambda,
                self.config.ppo.intrinsic_advantage_coefficient,
                self.config.ppo.normalize_advantages,
            )
            ppo_metrics = self.ppo.update(self.rollout_buffer, estimate)
            bigan_metrics = self.bigan.update(self.replay_buffer)
            if self.config.novelty.novelty_type == "transition":
                assert self.transition_trainer is not None
                assert self.transition_replay_buffer is not None
                transition_metrics = self.transition_trainer.update(
                    self.transition_replay_buffer,
                    self.config.bigan.batch_size,
                )
            else:
                transition_metrics = None
            metrics = TrainingIterationMetrics(
                update_index,
                self.environment_steps,
                summary["extrinsic"],
                summary["intrinsic"],
                summary["total"],
                summary["pixel"],
                summary["feature"],
                summary["normalized"],
                ppo_metrics,
                bigan_metrics,
            )
            results.append(metrics)
            if self.logger is not None:
                self.logger.log(
                    self.environment_steps,
                    {
                        "update": update_index,
                        "reward/extrinsic": metrics.extrinsic_reward,
                        "reward/intrinsic": metrics.intrinsic_reward,
                        "reward/total": metrics.total_reward,
                        "novelty/pixel": metrics.pixel_novelty,
                        "novelty/feature": metrics.feature_novelty,
                        "novelty/normalized": metrics.normalized_novelty,
                        "ppo": metrics.ppo,
                        "bigan": metrics.bigan,
                        "transition": metrics.transition,
                    },
                )
            if self.plotter is not None:
                self.plotter.maybe_plot(update_index)
        if self.plotter is not None:
            self.plotter.close()
        return results

    def close(self) -> None:
        """Close all environment, plotting, and logger resources.

        Input: This trainer.
        Output: No value; all resources are finalized.
        Mathematical meaning: Completes one reproducible vectorized run.
        """
        if self.plotter is not None:
            self.plotter.close()
        self.environment.close()
        if self.logger is not None:
            self.logger.close()
