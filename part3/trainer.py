"""Vectorized Part-3 training pipeline (causal representation attribution).

``Part3Trainer`` keeps every component of the system fixed and varies only the
observation representation ``E`` used to define the latent transition novelty::

    N_T(s_t, a_t, s_{t+1}) = || f(E(s_t), a_t) - E(s_{t+1}) ||_2

The primary causal comparisons are::

    E_BiGAN -> N_T -> PPO
    E_IDF   -> N_T -> PPO
    E_RND   -> N_T -> PPO

The environment, PPO, forward model, reward processing, budget, and seeds are
identical across the three arms; only ``E`` differs. Each ``E`` is trained
online from the same rollout stream with the same batch size, update frequency,
training-start condition, and replay capacity. There is no pretraining/freeze
schedule, no EME scaling, no episodic counts, no state novelty, and no
separate hyperparameter tuning.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import torch
from torch import Tensor

from agent.actor import ActorNetwork
from agent.advantage import TwoStreamAdvantageEstimate, compute_two_stream_returns_and_advantages
from agent.critic import DualCriticNetwork
from agent.ppo import PPOAgent, PPOUpdateMetrics
from agent.rollout_buffer import RolloutBuffer
from config import ExperimentConfig, Part3Config
from environments.vector_adapter import VectorEnvironmentAdapter
from exploration.intrinsic_reward import IntrinsicRewardPipeline
from exploration.transition_novelty import LatentForwardModel
from part3.representations import (
    RepresentationDescription,
    RepresentationUpdateMetrics,
    build_representation,
    describe_representation,
)
from part3.transition_novelty import (
    ForwardModelUpdateMetrics,
    Part3ForwardModelTrainer,
    Part3TransitionNoveltyEstimator,
)
from part3.rng import run_under_rng, RNG_FORWARD_MODEL_INIT, RNG_PPO_INIT
from utils.logger import ExperimentLogger
from utils.replay import ReplayBuffer
from utils.seed import derive_seed, seed_everything
from utils.transition_replay import TransitionReplayBuffer


@dataclass(frozen=True)
class Part3UpdateMetrics:
    """Diagnostics from one Part-3 rollout and update cycle."""

    update: int
    environment_steps: int
    extrinsic_reward: float
    intrinsic_reward: float
    total_reward: float
    transition_novelty: float
    normalized_novelty: float
    ppo: PPOUpdateMetrics
    representation: RepresentationUpdateMetrics
    forward_model: ForwardModelUpdateMetrics
    death_count: int
    death_fraction_high: float
    death_percentile: float


class Part3Trainer:
    """Run the causal representation-intervention experiment."""

    def __init__(
        self,
        environment: VectorEnvironmentAdapter,
        config: ExperimentConfig,
        logger: Optional[ExperimentLogger] = None,
    ) -> None:
        """Build the actor/critic, representation, forward model, and buffers.

        Input: Vector environment, complete configuration, and optional logger.
        Output: Initialized Part-3 trainer.
        Mathematical meaning: Instantiates the PPO agent, the representation
            ``E`` under test, and the shared forward model ``f``; the only
            varying component across runs is ``E``.
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

        observation_shape = config.environment.observation_shape
        action_dim = config.environment.action_dim
        discrete_actions = config.environment.discrete_actions
        part3: Part3Config = config.part3

        # PPO / actor / critic initialization is scoped to the PPO init stream so
        # it is isolated from representation and forward-model initialization.
        self.actor, self.critic = run_under_rng(
            config.seed.seed,
            RNG_PPO_INIT,
            lambda: (
                ActorNetwork(
                    observation_shape, action_dim, discrete_actions
                ),
                DualCriticNetwork(observation_shape),
            ),
        )
        self.actor = self.actor.to(self.device)
        self.critic = self.critic.to(self.device)

        samples_per_update = config.ppo.rollout_steps * self.num_envs
        total_updates = config.training.total_environment_steps // samples_per_update
        if total_updates <= 0:
            raise ValueError("training budget must contain at least one complete vector rollout")
        self.total_updates = total_updates
        self.ppo = PPOAgent(
            self.actor,
            self.critic,
            config.ppo,
            total_updates,
            beta=config.ppo.intrinsic_advantage_coefficient,
            device=self.device,
        )

        # Representation ``E`` under test (the only varying component).
        self.representation = build_representation(
            config.seed.seed,
            observation_shape,
            part3.representation,
            part3,
            action_dim,
            discrete_actions,
            self.device,
        )
        self.representation_description: RepresentationDescription = describe_representation(
            self.representation, part3
        )
        self.encoder = self.representation.encoder

        # Shared latent forward model ``f`` (identical for all representations).
        self.forward_model = run_under_rng(
            config.seed.seed,
            RNG_FORWARD_MODEL_INIT,
            lambda: LatentForwardModel(
                part3.latent_dim,
                action_dim,
                part3.forward_hidden_dim,
                discrete_actions,
            ),
        ).to(self.device)

        self.transition_novelty = Part3TransitionNoveltyEstimator(
            self.encoder,
            self.forward_model,
            part3.normalization_epsilon,
            part3.normalization_clip,
            self.device,
        )
        self.forward_trainer = Part3ForwardModelTrainer(
            self.transition_novelty,
            part3.forward_learning_rate,
            part3.forward_update_epochs,
            part3.forward_max_grad_norm,
        )
        self.forward_trainer.set_base_seed(config.seed.seed)
        self.reward_pipeline = IntrinsicRewardPipeline(config.novelty.clip_intrinsic_reward)

        # Same rollout/observation/transition stream and capacity for every arm.
        self.replay_capacity = part3.replay_capacity or max(samples_per_update, part3.batch_size)
        self.rollout_buffer = RolloutBuffer(
            config.ppo.rollout_steps,
            self.num_envs,
            observation_shape,
            () if discrete_actions else (action_dim,),
            self.device,
            torch.float32,
            torch.long if discrete_actions else torch.float32,
        )
        self.observation_replay = ReplayBuffer(
            self.replay_capacity,
            observation_shape,
            storage_device="cpu",
            observation_dtype=torch.float32,
        )
        self.transition_replay = TransitionReplayBuffer(
            self.replay_capacity,
            observation_shape,
            () if discrete_actions else (action_dim,),
            "cpu",
            torch.float32,
            torch.long if discrete_actions else torch.float32,
        )

        self._observation: Optional[Tensor] = None
        self.episode_returns = torch.zeros(self.num_envs, dtype=torch.float64)
        self.episode_lengths = torch.zeros(self.num_envs, dtype=torch.long)
        self.environment_steps = 0
        self.update_count = 0

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
        """Ensure a current batched observation is available.

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
        """Reset completed environment slots.

        Input: Terminal next-observation batch and boolean done mask ``[N]``.
        Output: Next current-observation batch with completed slots reset.
        Mathematical meaning: Starts new trajectories in the product MDP.
        """
        current = observations.clone()
        for index in torch.nonzero(done, as_tuple=False).flatten().tolist():
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
        Output: No value; completed scores are logged (if a logger exists).
        Mathematical meaning: Records empirical episodic extrinsic returns.
        """
        for index in torch.nonzero(done, as_tuple=False).flatten().tolist():
            score = float(self.episode_returns[index].item())
            length = int(self.episode_lengths[index].item())
            timestep = self.environment_steps + index
            if self.logger is not None:
                self.logger.log(
                    timestep,
                    {"episode/game_score": score, "episode/length": length},
                )

    def _collect_rollout(self) -> Dict[str, float]:
        """Collect T*N transitions, compute N_T and the intrinsic reward.

        Input: Current trainer state and configured rollout horizon T.
        Output: Mean reward and realism metrics across ``T*N`` samples plus
            death/respawn novelty diagnostics.
        Mathematical meaning: For each step, encodes ``s_t``/``s_{t+1}`` with
            ``E``, predicts the next code with ``f``, measures ``N_T``, applies
            the shared reward processing, and stores separate advantage streams.
        """
        self.rollout_buffer.reset()
        rollouts = int(self.config.ppo.rollout_steps)
        novelty_values: List[Tensor] = []
        done_masks: List[Tensor] = []
        death_novelty: List[Tensor] = []
        sums = {
            "extrinsic": 0.0,
            "intrinsic": 0.0,
            "total": 0.0,
            "novelty": 0.0,
            "normalized": 0.0,
        }
        for _ in range(rollouts):
            observations = self._reset_if_needed()
            actions, log_probabilities, extrinsic_values, intrinsic_values = self.ppo.select_action(observations)
            transition = self.environment.step(actions)
            next_observations = transition.observations.to(self.device)
            extrinsic_rewards = transition.rewards.to(self.device)
            novelty = self.transition_novelty(
                observations,
                actions,
                next_observations,
                extrinsic_rewards,
            )
            reward_result = self.reward_pipeline(extrinsic_rewards, novelty)
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
            self.observation_replay.add_batch(next_observations.detach().cpu())
            self.transition_replay.add_batch(
                observations.detach().cpu(),
                actions.detach().cpu(),
                next_observations.detach().cpu(),
            )
            done = transition.done
            novelty_values.append(novelty.novelty.detach().cpu())
            done_masks.append(done.cpu())
            if bool(done.any()):
                death_novelty.append(novelty.novelty.detach().cpu()[done.cpu()])
            self.episode_returns += extrinsic_rewards.detach().double().cpu()
            self.episode_lengths += 1
            self._record_completed_episodes(done.cpu())
            self.environment_steps += self.num_envs
            self._observation = self._reset_done(next_observations, done)
            sums["extrinsic"] += float(reward_result.extrinsic_reward.detach().sum().cpu())
            sums["intrinsic"] += float(reward_result.intrinsic_reward.detach().sum().cpu())
            sums["total"] += float((reward_result.extrinsic_reward + reward_result.intrinsic_reward).detach().sum().cpu())
            sums["novelty"] += float(novelty.novelty.detach().sum().cpu())
            sums["normalized"] += float(novelty.normalized_score.detach().sum().cpu())
        self.rollout_buffer.set_final_observation(self._reset_if_needed())

        denominator = float(rollouts * self.num_envs)
        all_novelty = torch.cat(novelty_values, dim=0)
        all_done = torch.cat(done_masks, dim=0)
        death_count = int((all_done).sum().item())
        death_fraction_high = 0.0
        death_percentile = 0.0
        if death_count > 0 and all_novelty.numel() > 0:
            death_values = torch.cat(death_novelty, dim=0)
            high_threshold = torch.quantile(all_novelty, 0.90)
            death_fraction_high = float((death_values >= high_threshold).float().mean())
            # Mean percentile rank of each death-transition novelty within the
            # rollout-wide novelty distribution (empirical CDF via searchsorted).
            sorted_novelty = torch.sort(all_novelty).values
            ranks = torch.searchsorted(
                sorted_novelty, death_values
            ).float() / float(all_novelty.numel())
            death_percentile = float(ranks.mean())
        return {
            "extrinsic": sums["extrinsic"] / denominator,
            "intrinsic": sums["intrinsic"] / denominator,
            "total": sums["total"] / denominator,
            "novelty": sums["novelty"] / denominator,
            "normalized": sums["normalized"] / denominator,
            "death_count": death_count,
            "death_fraction_high": death_fraction_high,
            "death_percentile": death_percentile,
        }

    def train(self) -> List[Part3UpdateMetrics]:
        """Run the Part-3 online training loop.

        Input: No arguments; uses immutable experiment configuration.
        Output: Per-update diagnostics.
        Mathematical meaning: Repeatedly collects ``T*N`` transitions, computes
            ``N_T``, runs two-stream GAE and PPO, then updates the
            representation and the shared forward model online.
        """
        results: List[Part3UpdateMetrics] = []
        for update_index in range(self.total_updates):
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
            # Online representation update (same batch size, cadence, and
            # training-start condition for the three arms).
            if self.config.part3.representation == "idf":
                representation_metrics = self.representation.train(self.transition_replay)
            else:
                representation_metrics = self.representation.train(self.observation_replay)
            forward_metrics = self.forward_trainer.update(
                self.transition_replay, self.config.part3.batch_size
            )
            self.update_count += 1
            metrics = Part3UpdateMetrics(
                update_index,
                self.environment_steps,
                summary["extrinsic"],
                summary["intrinsic"],
                summary["total"],
                summary["novelty"],
                summary["normalized"],
                ppo_metrics,
                representation_metrics,
                forward_metrics,
                summary["death_count"],
                summary["death_fraction_high"],
                summary["death_percentile"],
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
                        "novelty/NT": metrics.transition_novelty,
                        "novelty/normalized": metrics.normalized_novelty,
                        "representation/loss": metrics.representation.loss,
                        "representation/updated": float(metrics.representation.updated),
                        "forward_model/mse": metrics.forward_model.mse_loss,
                        "forward_model/updated": float(metrics.forward_model.updated),
                        "death/count": metrics.death_count,
                        "death/fraction_high": metrics.death_fraction_high,
                        "death/novelty_percentile": metrics.death_percentile,
                        "ppo": metrics.ppo,
                    },
                )
        return results

    def close(self) -> None:
        """Close the environment and logger resources.

        Input: This trainer.
        Output: No value; resources are finalized.
        Mathematical meaning: Completes one reproducible vectorized run.
        """
        self.environment.close()
        if self.logger is not None:
            self.logger.close()
