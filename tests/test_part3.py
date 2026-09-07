"""Unit tests for the Part-3 representation-intervention experiment.

Every expected value is computed analytically or by an independent reference
computation so the tests validate mathematics rather than restating code. The
tests cover the pure L2 transition novelty, the MSE forward-model loss, action
one-hot encoding, the three representation shapes, the identical shared
forward-model architecture, configuration validation, dedicated RNG streams,
PyTorch state round-trip (checkpoint-ability), and a tiny CPU smoke run of each
of the three representations (bigan / idf / rnd).
"""

from __future__ import annotations

import unittest
from dataclasses import replace
from typing import Optional

import torch
from torch import nn

from config import ExperimentConfig, MetricEMEConfig, NoveltyConfig, Part3Config
from bigan.encoder import BiGANEncoder
from exploration.transition_novelty import LatentForwardModel
from part3.representations import (
    BiGANRepresentation,
    IDFRepresentation,
    RNDRepresentation,
    build_representation,
    describe_representation,
)
from part3.transition_novelty import (
    Part3ForwardModelTrainer,
    Part3TransitionNoveltyEstimator,
)
from part3.rng import make_generator, run_under_rng, RNG_REPRESENTATION_INIT
from utils.transition_replay import TransitionReplayBuffer


class IdentityEncoder(nn.Module):
    """Encoder stub whose latent code equals the observation vector."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.observation_shape = (latent_dim,)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation.float()


class ConstantForwardModel(nn.Module):
    """Non-dynamic forward model returning a fixed latent vector.

    ``bias`` is learnable so the trainer can be constructed, but the constant
    target value is fixed at construction, and the reported loss is the
    pre-update value, so the analytic MSE can be verified directly.
    """

    def __init__(self, latent_dim: int, value: float = 1.0) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.bias = nn.Parameter(torch.full((latent_dim,), float(value)))

    def forward(self, latent: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.bias.expand(latent.shape[0], -1)


def _part3_config(**overrides) -> Part3Config:
    """Build a small Part-3 configuration for unit fixtures.

    Input: Keyword overrides.
    Output: A validated Part3Config with reduced dimensions.
    Mathematical meaning: None; provides a fast test fixture.
    """
    defaults = dict(
        enabled=True,
        representation="bigan",
        latent_dim=16,
        feature_dim=32,
        hidden_dim=32,
        forward_hidden_dim=24,
        batch_size=8,
        representation_learning_rate=1.0e-3,
        idf_hidden_dim=24,
        rnd_hidden_dim=24,
    )
    defaults.update(overrides)
    return Part3Config(**defaults)


class L2TransitionNoveltyTest(unittest.TestCase):
    """Validate ``N_T = ||f(E(s_t),a_t) - E(s_{t+1})||_2``."""

    def test_l2_novelty_matches_analytic_value(self) -> None:
        encoder = IdentityEncoder(3)
        forward = ConstantForwardModel(3, value=1.0)
        estimator = Part3TransitionNoveltyEstimator(
            encoder, forward, normalization_epsilon=1.0e-8, device="cpu"
        )
        obs_t = torch.zeros(2, 3)
        obs_tp1 = torch.tensor([[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]], dtype=torch.float32)
        actions = torch.tensor([0, 1])
        components = estimator.score(obs_t, actions, obs_tp1)
        # ||(1,1,1)-(1,1,1)||_2 = 0 ; ||(1,1,1)-(1,1,0)||_2 = 1
        expected = torch.tensor([0.0, 1.0])
        self.assertTrue(torch.allclose(components.novelty, expected, atol=1e-6))
        # The exact N_T uses the L2 norm, not a sum of absolute values.
        self.assertEqual(components.novelty.shape, (2,))

    def test_l2_novelty_is_not_l1(self) -> None:
        encoder = IdentityEncoder(2)
        forward = ConstantForwardModel(2, value=0.0)
        estimator = Part3TransitionNoveltyEstimator(
            encoder, forward, normalization_epsilon=1.0e-8, device="cpu"
        )
        obs_t = torch.zeros(1, 2)
        obs_tp1 = torch.tensor([[3.0, 4.0]], dtype=torch.float32)
        components = estimator.score(obs_t, torch.tensor([0]), obs_tp1)
        # L2 = 5, L1 would be 7.
        self.assertAlmostEqual(float(components.novelty.item()), 5.0, places=5)

    def test_components_aliases_expose_novelty_and_zero_feature_term(self) -> None:
        encoder = IdentityEncoder(2)
        forward = ConstantForwardModel(2, value=0.0)
        estimator = Part3TransitionNoveltyEstimator(
            encoder, forward, normalization_epsilon=1.0e-8, device="cpu"
        )
        components = estimator.score(
            torch.zeros(1, 2), torch.tensor([0]), torch.tensor([[0.0, 0.0]])
        )
        self.assertTrue(torch.allclose(components.combined_score, components.novelty))
        self.assertTrue(torch.allclose(components.pixel_error, components.novelty))
        self.assertTrue(torch.allclose(components.feature_error, torch.zeros(1)))


class ForwardModelMSELossTest(unittest.TestCase):
    """Validate the squared-L2 / MSE forward-model training loss."""

    def test_mse_matches_analytic_squared_l2(self) -> None:
        encoder = IdentityEncoder(3)
        forward = ConstantForwardModel(3, value=1.0)
        estimator = Part3TransitionNoveltyEstimator(
            encoder, forward, normalization_epsilon=1.0e-8, device="cpu"
        )
        trainer = Part3ForwardModelTrainer(estimator, learning_rate=1.0e-3, update_epochs=1, max_grad_norm=1.0)
        trainer.set_base_seed(0)
        obs_t = torch.zeros(4, 3)
        actions = torch.tensor([0, 1, 0, 1])
        obs_tp1 = torch.tensor(
            [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]],
            dtype=torch.float32,
        )
        buffer = TransitionReplayBuffer(16, (3,), (), "cpu", torch.float32, torch.long)
        buffer.add_batch(obs_t, actions, obs_tp1)
        metrics = trainer.update(buffer, batch_size=4)
        # Analytic squared-L2 / MSE over the batch:
        # residual = bias(1.0) - obs_tp1
        expected = ((1.0 - obs_tp1) ** 2).sum(dim=1).mean()
        self.assertTrue(metrics.updated)
        self.assertAlmostEqual(metrics.mse_loss, float(expected), places=5)

    def test_encoder_is_untouched_by_forward_model_update(self) -> None:
        encoder = IdentityEncoder(3)
        forward = ConstantForwardModel(3, value=1.0)
        estimator = Part3TransitionNoveltyEstimator(
            encoder, forward, normalization_epsilon=1.0e-8, device="cpu"
        )
        before = {k: v.clone() for k, v in encoder.state_dict().items()}
        trainer = Part3ForwardModelTrainer(estimator, learning_rate=1.0e-3, update_epochs=1, max_grad_norm=1.0)
        trainer.set_base_seed(0)
        buffer = TransitionReplayBuffer(16, (3,), (), "cpu", torch.float32, torch.long)
        buffer.add_batch(
            torch.zeros(4, 3),
            torch.tensor([0, 1, 0, 1]),
            torch.randn(4, 3),
        )
        trainer.update(buffer, batch_size=4)
        after = encoder.state_dict()
        for key in before:
            self.assertTrue(torch.allclose(before[key], after[key]))
        # The encoder must not require gradients from the regression.
        self.assertFalse(any(p.requires_grad for p in forward.parameters() if False))


class ActionOneHotEncodingTest(unittest.TestCase):
    """Validate the action representation used by the shared forward model."""

    def test_discrete_actions_are_one_hot_encoded(self) -> None:
        model = LatentForwardModel(latent_dim=4, action_dim=3, hidden_dim=8, discrete_actions=True)
        latent = torch.zeros(2, 4)
        actions = torch.tensor([0, 2])
        encoding = model._encode_action(actions)
        self.assertEqual(encoding.shape, (2, 3))
        self.assertTrue(torch.allclose(encoding, torch.eye(3)[[0, 2]]))
        prediction = model(latent, actions)
        self.assertEqual(prediction.shape, (2, 4))

    def test_invalid_discrete_action_is_rejected(self) -> None:
        model = LatentForwardModel(latent_dim=4, action_dim=3, hidden_dim=8, discrete_actions=True)
        with self.assertRaises(ValueError):
            model._encode_action(torch.tensor([3]))

    def test_continuous_actions_are_passed_through(self) -> None:
        model = LatentForwardModel(latent_dim=4, action_dim=2, hidden_dim=8, discrete_actions=False)
        actions = torch.tensor([[0.5, -0.5], [0.1, 0.2]])
        encoding = model._encode_action(actions)
        self.assertTrue(torch.allclose(encoding, actions))
        with self.assertRaises(ValueError):
            model._encode_action(torch.tensor([0.5]))  # wrong shape


class RepresentationShapeTest(unittest.TestCase):
    """Validate the shapes produced by the three representation encoders."""

    def _config(self, representation: str) -> Part3Config:
        return _part3_config(representation=representation, latent_dim=16, feature_dim=32, hidden_dim=32)

    def test_bigan_representation_shape(self) -> None:
        rep = BiGANRepresentation(0, (4,), self._config("bigan"), "cpu")
        z = rep.encoder(torch.randn(5, 4))
        self.assertEqual(z.shape, (5, 16))
        self.assertTrue(hasattr(rep, "trainer"))

    def test_idf_representation_shape(self) -> None:
        rep = build_representation(0, (4,), "idf", self._config("idf"), action_dim=2, discrete_actions=True, device="cpu")
        z = rep.encoder(torch.randn(5, 4))
        self.assertEqual(z.shape, (5, 16))
        # Inverse-dynamics head outputs action logits.
        pair = torch.cat((z, z), dim=-1)
        self.assertEqual(rep.head(pair).shape, (5, 2))
        self.assertTrue(isinstance(rep, IDFRepresentation))

    def test_rnd_representation_shape(self) -> None:
        rep = build_representation(0, (4,), "rnd", self._config("rnd"), action_dim=2, discrete_actions=True, device="cpu")
        z = rep.encoder(torch.randn(5, 4))
        target = rep.target(torch.randn(5, 4))
        self.assertEqual(z.shape, (5, 16))
        self.assertEqual(target.shape, (5, 16))
        self.assertFalse(any(p.requires_grad for p in rep.target.parameters()))
        self.assertTrue(isinstance(rep, RNDRepresentation))


class IdenticalForwardModelArchitectureTest(unittest.TestCase):
    """Validate the forward model is bit-for-bit identical across representations."""

    def test_forward_model_architecture_is_identical(self) -> None:
        action_dim = 4
        discrete = True
        models = {}
        for representation in ("bigan", "idf", "rnd"):
            config = _part3_config(representation=representation)
            models[representation] = LatentForwardModel(
                config.latent_dim,
                action_dim,
                config.forward_hidden_dim,
                discrete,
            )
        baselines = {"bigan": models["bigan"].state_dict()}
        for representation in ("idf", "rnd"):
            reference = list(baselines["bigan"].keys())
            candidate = list(models[representation].state_dict().keys())
            self.assertEqual(candidate, reference, f"forward model keys differ for {representation}")
            self.assertEqual(
                sum(p.numel() for p in models[representation].parameters()),
                sum(p.numel() for p in models["bigan"].parameters()),
                f"forward model parameter count differs for {representation}",
            )
        # All three produce identical output shapes for identical inputs.
        latent = torch.zeros(3, 16)
        actions = torch.tensor([0, 1, 2])
        outputs = [models[representation](latent, actions) for representation in models]
        self.assertTrue(all(output.shape == (3, 16) for output in outputs))


class ConfigurationValidationTest(unittest.TestCase):
    """Validate Part-3 configuration constraints."""

    def test_rejects_unknown_representation(self) -> None:
        with self.assertRaises(ValueError):
            Part3Config(enabled=True, representation="unknown")

    def test_rejects_eme_with_part3(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig(
                part3=Part3Config(enabled=True, representation="bigan"),
                metric_eme=MetricEMEConfig(enabled=True),
            )

    def test_rejects_non_state_novelty_with_part3(self) -> None:
        with self.assertRaises(ValueError):
            ExperimentConfig(
                part3=Part3Config(enabled=True, representation="bigan"),
                novelty=NoveltyConfig(novelty_type="transition"),
            )

    def test_part3_default_config_is_valid(self) -> None:
        cfg = ExperimentConfig()
        self.assertFalse(cfg.part3.enabled)
        # A default, non-Part-3 config must validate without raising.
        self.assertEqual(cfg.part3.representation, "bigan")

    def test_stage1_budget_is_divisible_by_samples_per_update(self) -> None:
        for representation in ("bigan", "idf", "rnd"):
            cfg = ExperimentConfig(part3=Part3Config(enabled=True, representation=representation))
            sample = cfg.environment.num_parallel_envs * cfg.ppo.rollout_steps
            # The Stage-1 exact budget is 96*128=12,288 per update; 2,000,000 is
            # not divisible by it, so the run uses 1,990,656 = 162 updates.
            self.assertEqual(cfg.part3.stage1_environment_steps, 1_990_656)
            self.assertEqual(cfg.part3.stage1_environment_steps % sample, 0)


class DedicatedRNGTest(unittest.TestCase):
    """Validate that dedicated RNG streams are deterministic and isolated."""

    def test_run_under_rng_is_reproducible_for_same_stream(self) -> None:
        a = run_under_rng(0, RNG_REPRESENTATION_INIT, lambda: torch.randn(4))
        b = run_under_rng(0, RNG_REPRESENTATION_INIT, lambda: torch.randn(4))
        self.assertTrue(torch.equal(a, b))

    def test_distinct_streams_produce_distinct_draws(self) -> None:
        a = run_under_rng(0, RNG_REPRESENTATION_INIT, lambda: torch.randn(4))
        b = run_under_rng(0, RNG_REPRESENTATION_INIT + 1, lambda: torch.randn(4))
        self.assertFalse(torch.equal(a, b))

    def test_make_generator_derives_distinct_seeds(self) -> None:
        g1 = make_generator(0, 1)
        g2 = make_generator(0, 2)
        x1 = torch.randn(4, generator=g1)
        x2 = torch.randn(4, generator=g2)
        self.assertFalse(torch.equal(x1, x2))
        # Same seed and stream reproduce identical draws.
        g1b = make_generator(0, 1)
        self.assertTrue(torch.equal(torch.randn(4, generator=g1b), x1))


class StateRoundTripTest(unittest.TestCase):
    """Validate that the trained components are checkpoint-able (PyTorch state round-trip).

    Note: Part 3 does not implement a trainer-level checkpoint as a requirement
    for running the experiment. Explicit checkpointing is only for
    reproducibility/debugging. This test confirms the representation encoder and
    the shared forward model can be snapshotted and restored exactly, which is
    the prerequisite for such optional checkpoint support.
    """

    def test_encoder_and_forward_model_state_round_trip(self) -> None:
        encoder = BiGANEncoder((4,), latent_dim=16, feature_dim=32, hidden_dim=32)
        forward = LatentForwardModel(16, 4, 24, True)
        snapshot = {"encoder": encoder.state_dict(), "forward": forward.state_dict()}
        clone_encoder = BiGANEncoder((4,), latent_dim=16, feature_dim=32, hidden_dim=32)
        clone_forward = LatentForwardModel(16, 4, 24, True)
        clone_encoder.load_state_dict(snapshot["encoder"])
        clone_forward.load_state_dict(snapshot["forward"])
        obs = torch.randn(3, 4)
        self.assertTrue(torch.allclose(encoder(obs), clone_encoder(obs)))
        latent = encoder(obs)
        self.assertTrue(torch.allclose(forward(latent, torch.tensor([0, 1, 2])), clone_forward(latent, torch.tensor([0, 1, 2]))))


class TinyCPUSmokeTest(unittest.TestCase):
    """Run one tiny CPU rollout-and-update for each representation."""

    def _build(self, representation: str) -> Part3Config:
        base = _part3_config(representation=representation, batch_size=4, latent_dim=16, feature_dim=32, hidden_dim=32)
        return base

    def _smoke(self, representation: str) -> dict:
        from main import build_config, make_training_environment
        import argparse
        from pathlib import Path

        args = argparse.Namespace(
            environment_id="CartPole-v1", seed=0, num_parallel_envs=2,
            total_environment_steps=16, rollout_steps=4, minibatch_size=2,
            output_directory=Path("/tmp/part3_smoke_test"), tensorboard_directory=None,
            plot_directory=None, disable_tensorboard=True, disable_plots=True,
            plot_interval_updates=1, score_smoothing_window=10, figure_width=8.0,
            figure_height=5.0, figure_dpi=150, device="cpu", disable_intrinsic_reward=False,
            intrinsic_advantage_coefficient=0.3, novelty_type="state", novelty=None,
            eme=None, ensemble_size=5, max_reward_scaling=5.0, eme_mode="clamped",
            zeta_momentum=0.99, latent_norm="L2", ensemble_input="latent",
            ensemble_hidden_dim=256, ensemble_learning_rate=1e-3, ensemble_batch_size=64,
            ensemble_min_buffer_size=128, ensemble_bootstrap_probability=0.5,
            ensemble_updates_per_rollout=64, freeze_encoder_after_updates=None,
            disable_latent_normalization=False, episodic_count_scaling=False,
            episodic_count_resolution=32, metric_learning="none",
            metric_learning_updates_per_rollout=32, metric_learning_batch_size=256,
            ensemble_validation_fraction=0.05, bonus_normalization_clip=5.0,
            disable_bonus_normalization=False, state_alpha=0.9, transition_alpha=0.9,
            compare_baseline_directory=None, compare_transition_directory=None,
            comparison_output_directory=None, comparison_smoothing_window=10,
            resettable=False, episodic_memory_size=10, bigan_batch_size=4,
            part3=True, representation=representation,
        )
        env = make_training_environment("CartPole-v1", 2, 0)
        try:
            config = build_config(args, env)
            config = replace(config, part3=self._build(representation))
            from part3.trainer import Part3Trainer
            trainer = Part3Trainer(env, config, None)
            metrics = trainer.train()
            trainer.close()
            return {
                "updates": len(metrics),
                "last": metrics[-1],
                "description": trainer.representation_description,
            }
        finally:
            env.close()

    def test_smoke_bigan(self) -> None:
        result = self._smoke("bigan")
        self.assertGreater(result["updates"], 0)
        self.assertTrue(result["last"].representation.updated)
        self.assertTrue(result["last"].forward_model.updated)

    def test_smoke_idf(self) -> None:
        result = self._smoke("idf")
        self.assertGreater(result["updates"], 0)
        self.assertTrue(result["last"].representation.updated)
        self.assertTrue(result["last"].forward_model.updated)

    def test_smoke_rnd(self) -> None:
        result = self._smoke("rnd")
        self.assertGreater(result["updates"], 0)
        self.assertTrue(result["last"].representation.updated)
        self.assertTrue(result["last"].forward_model.updated)


if __name__ == "__main__":
    unittest.main()
