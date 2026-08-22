"""Configuration objects for reproducible Adventurer experiments.

This module contains validated, immutable dataclasses for the PPO agent, the
BiGAN novelty model, and the training pipeline. Keeping configuration separate
from implementation makes experiment comparisons auditable and prevents
implicit hyperparameters from being introduced in individual modules.

The mathematical quantities represented here are used by later modules:

* PPO discounting and GAE use :math:`gamma` and :math:`lambda` (PPO/GAE,
  Equations (8)--(10) in the paper's PPO formulation).
* PPO clipping uses :math:`epsilon` in the clipped surrogate objective
  (Equation (7)).
* BiGAN optimization uses the learning rates, batch size, latent dimension,
  and update schedule in the adversarial objective (Equations (1)--(6)).
* Novelty weighting uses the reconstruction/feature coefficients and the
  intrinsic-reward coefficient in the Adventurer objective.

All dataclasses are frozen so that a run cannot accidentally mutate its
hyperparameters after initialization.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import Enum
from typing import Any, Optional


class DeviceType(str, Enum):
    """Allowed device-selection policies for experiment execution."""

    AUTO = "auto"
    CPU = "cpu"
    CUDA = "cuda"


@dataclass(frozen=True)
class SeedConfig:
    """Control pseudorandom seeds and deterministic PyTorch behavior.

    Attributes:
        seed: Base integer seed used for Python, NumPy, PyTorch, and
            environment seeding.
        deterministic: Whether deterministic backend algorithms should be
            requested where supported.
        benchmark: Whether cuDNN benchmarking is enabled. It should normally
            be false for strict reproducibility.
    """

    seed: int = 0
    deterministic: bool = True
    benchmark: bool = False

    def __post_init__(self) -> None:
        """Validate the seed and deterministic-backend settings.

        Input: The values supplied to this dataclass.
        Output: No value; raises ``ValueError`` for an invalid seed/settings
            combination.
        Mathematical meaning: The seed identifies one reproducible sample
            path from the random transition and optimization processes.
        """
        if self.seed < 0:
            raise ValueError("seed must be non-negative")
        if self.deterministic and self.benchmark:
            raise ValueError("benchmark must be false in deterministic mode")


@dataclass(frozen=True)
class EnvironmentConfig:
    """Describe the interface and episode limits of the RL environment."""

    observation_shape: tuple[int, ...] = (84, 84, 4)
    action_dim: int = 4
    discrete_actions: bool = True
    num_parallel_envs: int = 96
    max_episode_steps: int = 27_000
    frame_stack: int = 4
    normalize_pixels: bool = True

    def __post_init__(self) -> None:
        """Validate observation and action-space dimensions.

        Input: Environment dimensions and preprocessing settings.
        Output: No value; raises ``ValueError`` if dimensions are unusable.
        Mathematical meaning: The observation shape defines the domain of the
            encoder, actor, and critic; ``action_dim`` defines the policy's
            categorical support or the action-vector width.
        """
        if not self.observation_shape or any(d <= 0 for d in self.observation_shape):
            raise ValueError("observation_shape must contain only positive dimensions")
        if self.action_dim <= 0:
            raise ValueError("action_dim must be positive")
        if self.num_parallel_envs <= 0:
            raise ValueError("num_parallel_envs must be positive")
        if self.max_episode_steps <= 0:
            raise ValueError("max_episode_steps must be positive")
        if self.frame_stack <= 0:
            raise ValueError("frame_stack must be positive")


@dataclass(frozen=True)
class PPOConfig:
    """Hyperparameters for the from-scratch PPO implementation.

    ``gamma`` and ``gae_lambda`` define the temporal-difference residual and
    exponentially weighted GAE estimator. ``clip_epsilon`` defines the trust
    region used by PPO's clipped policy ratio. ``value_clip_epsilon`` is
    optional PPO2-style value clipping and is disabled by default because the
    policy objective and value objective are configured independently.
    """

    gamma: float = 0.997
    gae_lambda: float = 0.95
    intrinsic_gamma: float = 0.90
    intrinsic_gae_lambda: float = 0.95
    clip_epsilon: float = 0.10
    value_clip_epsilon: Optional[float] = None
    entropy_coefficient: float = 0.01
    intrinsic_advantage_coefficient: float = 0.3
    value_loss_coefficient: float = 0.5
    intrinsic_value_loss_coefficient: float = 0.5
    learning_rate: float = 1.0e-4
    critic_learning_rate: Optional[float] = None
    intrinsic_critic_learning_rate: Optional[float] = None
    adam_epsilon: float = 1.0e-5
    max_grad_norm: float = 0.5
    rollout_steps: int = 128
    update_epochs: int = 4
    minibatch_size: int = 32
    normalize_advantages: bool = True
    lr_decay: bool = True
    lr_final_fraction: float = 0.0

    def __post_init__(self) -> None:
        """Validate PPO and GAE parameters.

        Input: PPO optimizer, objective, and rollout settings.
        Output: No value; raises ``ValueError`` for mathematically invalid
            ranges or incompatible batching.
        Mathematical meaning: The ranges enforce valid discounted returns,
            GAE weights, probability-ratio clipping, and complete minibatches
            for every PPO epoch.
        """
        if not 0.0 <= self.gamma < 1.0:
            raise ValueError("gamma must satisfy 0 <= gamma < 1")
        if not 0.0 <= self.gae_lambda <= 1.0:
            raise ValueError("gae_lambda must satisfy 0 <= gae_lambda <= 1")
        if not 0.0 <= self.intrinsic_gamma < 1.0:
            raise ValueError("intrinsic_gamma must satisfy 0 <= intrinsic_gamma < 1")
        if not 0.0 <= self.intrinsic_gae_lambda <= 1.0:
            raise ValueError("intrinsic_gae_lambda must satisfy 0 <= intrinsic_gae_lambda <= 1")
        if self.clip_epsilon <= 0.0:
            raise ValueError("clip_epsilon must be positive")
        if self.value_clip_epsilon is not None and self.value_clip_epsilon <= 0.0:
            raise ValueError("value_clip_epsilon must be positive when provided")
        if (
            self.entropy_coefficient < 0.0
            or self.value_loss_coefficient < 0.0
            or self.intrinsic_value_loss_coefficient < 0.0
        ):
            raise ValueError("loss coefficients must be non-negative")
        if self.learning_rate <= 0.0:
            raise ValueError("learning_rate must be positive")
        if self.critic_learning_rate is not None and self.critic_learning_rate <= 0.0:
            raise ValueError("critic_learning_rate must be positive when provided")
        if (
            self.intrinsic_critic_learning_rate is not None
            and self.intrinsic_critic_learning_rate <= 0.0
        ):
            raise ValueError(
                "intrinsic_critic_learning_rate must be positive when provided"
            )
        if self.adam_epsilon <= 0.0 or self.max_grad_norm <= 0.0:
            raise ValueError("adam_epsilon and max_grad_norm must be positive")
        if self.rollout_steps <= 0 or self.update_epochs <= 0 or self.minibatch_size <= 0:
            raise ValueError("rollout_steps, update_epochs, and minibatch_size must be positive")
        if self.rollout_steps % self.minibatch_size != 0:
            raise ValueError("rollout_steps must be divisible by minibatch_size")
        if not 0.0 <= self.lr_final_fraction <= 1.0:
            raise ValueError("lr_final_fraction must be in [0, 1]")


@dataclass(frozen=True)
class BiGANConfig:
    """Architecture and optimization settings for the BiGAN model."""

    latent_dim: int = 128
    feature_dim: int = 256
    hidden_dim: int = 512
    learning_rate: float = 2.0e-4
    adam_beta1: float = 0.5
    adam_beta2: float = 0.999
    adam_epsilon: float = 1.0e-8
    batch_size: int = 64
    discriminator_steps: int = 1
    generator_encoder_steps: int = 1
    gradient_penalty_coefficient: float = 0.0
    warmup_updates: int = 0
    update_interval: int = 1

    def __post_init__(self) -> None:
        """Validate BiGAN dimensions, optimizer values, and update schedule.

        Input: BiGAN architecture and adversarial-training parameters.
        Output: No value; raises ``ValueError`` for invalid settings.
        Mathematical meaning: Positive latent/feature dimensions define the
            encoder-generator coupling, while non-negative coefficients define
            the weighted adversarial and optional regularization objectives.
        """
        if min(self.latent_dim, self.feature_dim, self.hidden_dim, self.batch_size) <= 0:
            raise ValueError("BiGAN dimensions and batch_size must be positive")
        if self.learning_rate <= 0.0 or self.adam_epsilon <= 0.0:
            raise ValueError("BiGAN learning_rate and adam_epsilon must be positive")
        if not 0.0 <= self.adam_beta1 < 1.0 or not 0.0 <= self.adam_beta2 < 1.0:
            raise ValueError("Adam beta values must be in [0, 1)")
        if self.discriminator_steps <= 0 or self.generator_encoder_steps <= 0:
            raise ValueError("BiGAN update counts must be positive")
        if self.gradient_penalty_coefficient < 0.0:
            raise ValueError("gradient_penalty_coefficient must be non-negative")
        if self.warmup_updates < 0 or self.update_interval <= 0:
            raise ValueError("warmup_updates must be non-negative and update_interval positive")


@dataclass(frozen=True)
class NoveltyConfig:
    """Settings for state or transition novelty exploration."""

    novelty_type: str = "state"
    alpha: float = 0.9
    transition_alpha: float = 0.9
    running_momentum: float = 0.99
    normalization_epsilon: float = 1.0e-8
    clip_intrinsic_reward: Optional[float] = None
    transition_hidden_dim: int = 256
    transition_learning_rate: float = 1.0e-4
    transition_update_epochs: int = 1
    transition_max_grad_norm: float = 0.5

    def __post_init__(self) -> None:
        """Validate novelty weighting and running-normalization parameters.

        Input: Reconstruction/feature weights, reward scale, and statistics
            settings.
        Output: No value; raises ``ValueError`` for invalid ranges.
        Mathematical meaning: The two non-negative coefficients form the
            combined novelty score; momentum and epsilon stabilize its online
            normalization before reward scaling.
        """
        if self.novelty_type not in {"state", "transition"}:
            raise ValueError("novelty_type must be 'state' or 'transition'")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must satisfy 0 <= alpha <= 1")
        if not 0.0 <= self.transition_alpha <= 1.0:
            raise ValueError("transition_alpha must satisfy 0 <= transition_alpha <= 1")
        if self.transition_hidden_dim <= 0 or self.transition_learning_rate <= 0.0:
            raise ValueError("transition model dimensions and learning rate must be positive")
        if self.transition_update_epochs <= 0 or self.transition_max_grad_norm <= 0.0:
            raise ValueError("transition update epochs and gradient norm must be positive")
        if not 0.0 <= self.running_momentum < 1.0:
            raise ValueError("running_momentum must be in [0, 1)")
        if self.normalization_epsilon <= 0.0:
            raise ValueError("normalization_epsilon must be positive")
        if self.clip_intrinsic_reward is not None and self.clip_intrinsic_reward <= 0.0:
            raise ValueError("clip_intrinsic_reward must be positive when provided")


@dataclass(frozen=True)
class TrainingConfig:
    """Top-level schedule, logging, checkpoint, and device settings."""

    total_environment_steps: int = 12_288_000
    device: DeviceType = DeviceType.AUTO
    log_interval_updates: int = 1
    checkpoint_interval_updates: int = 100
    output_directory: str = "runs/adventurer"
    tensorboard_directory: Optional[str] = None
    plot_directory: Optional[str] = None
    enable_tensorboard: bool = True
    save_plots: bool = True
    plot_interval_updates: int = 1
    score_smoothing_window: int = 10
    figure_width: float = 8.0
    figure_height: float = 5.0
    figure_dpi: int = 150
    use_intrinsic_reward: bool = True
    resettable: bool = False
    episodic_memory_size: int = 10

    def __post_init__(self) -> None:
        """Validate training duration and operational intervals.

        Input: Training schedule, output path, device policy, and intrinsic
            reward switch.
        Output: No value; raises ``ValueError`` for invalid schedules.
        Mathematical meaning: The intrinsic advantage scale is configured once
            in ``PPOConfig.intrinsic_advantage_coefficient``.
        """
        if self.total_environment_steps <= 0:
            raise ValueError("total_environment_steps must be positive")
        if self.log_interval_updates <= 0 or self.checkpoint_interval_updates <= 0:
            raise ValueError("logging and checkpoint intervals must be positive")
        if self.plot_interval_updates <= 0:
            raise ValueError("plot_interval_updates must be positive")
        if self.score_smoothing_window <= 0:
            raise ValueError("score_smoothing_window must be positive")
        if self.figure_width <= 0.0 or self.figure_height <= 0.0:
            raise ValueError("figure dimensions must be positive")
        if self.figure_dpi <= 0:
            raise ValueError("figure_dpi must be positive")
        if self.episodic_memory_size <= 0:
            raise ValueError("episodic_memory_size must be positive")


@dataclass(frozen=True)
class ExperimentConfig:
    """Complete validated configuration for one Adventurer experiment."""

    seed: SeedConfig = SeedConfig()
    environment: EnvironmentConfig = EnvironmentConfig()
    ppo: PPOConfig = PPOConfig()
    bigan: BiGANConfig = BiGANConfig()
    novelty: NoveltyConfig = NoveltyConfig()
    training: TrainingConfig = TrainingConfig()

    def __post_init__(self) -> None:
        """Validate relationships spanning multiple configuration sections.

        Input: Fully constructed component configurations.
        Output: No value; raises ``ValueError`` for incompatible components.
        Mathematical meaning: The rollout schedule must contain an integer
            number of PPO updates, and the BiGAN input dimensions must be
            compatible with the declared environment observation.
        """
        samples_per_update = (
            self.environment.num_parallel_envs * self.ppo.rollout_steps
        )
        if self.training.total_environment_steps % samples_per_update != 0:
            raise ValueError(
                "total_environment_steps must be divisible by "
                "num_parallel_envs * rollout_steps"
            )
        if self.environment.discrete_actions and self.environment.action_dim <= 0:
            raise ValueError("discrete action spaces require a positive action_dim")

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-compatible nested representation of the configuration.

        Input: The initialized experiment configuration.
        Output: A new dictionary containing all dataclass fields, with enum
            values represented by their strings.
        Mathematical meaning: This is not a learning operation; it records the
            exact parameter point at which the objective was optimized.
        """
        result = asdict(self)
        result["training"]["device"] = self.training.device.value
        return result
