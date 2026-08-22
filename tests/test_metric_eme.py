"""Unit tests for the metric-based exploration bonus (Contribution 2).

Every expected value is computed analytically or by an independent reference
computation so the tests validate mathematics rather than restating code.
"""

from __future__ import annotations

import unittest

import torch

from config import ExperimentConfig, MetricEMEConfig
from bigan.encoder import BiGANEncoder
from exploration.ensemble_scaling import EnsembleRewardVariance, RewardNet
from exploration.intrinsic_reward import MetricIntrinsicReward
from exploration.state_discrepancy import LatentStateDiscrepancy


class IdentityEncoder(torch.nn.Module):
    """Encoder stub whose latent code equals the observation vector."""

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.observation_shape = (latent_dim,)

    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation.float()


class LatentStateDiscrepancyTest(unittest.TestCase):
    """Validate the latent metric term ``||E(s_t)-E(s_{t+1})||_p``."""

    def test_l2_distance_matches_analytic_value(self) -> None:
        discrepancy = LatentStateDiscrepancy(IdentityEncoder(3), norm="L2")
        obs_t = torch.tensor([[0.0, 0.0, 0.0], [1.0, 2.0, 2.0]])
        obs_tp1 = torch.tensor([[3.0, 4.0, 0.0], [1.0, 2.0, 2.0]])
        distances = discrepancy(obs_t, obs_tp1)
        # ||(3,4,0)||_2 = 5 exactly; identical states have zero distance.
        self.assertTrue(torch.allclose(distances, torch.tensor([5.0, 0.0])))

    def test_l1_distance_matches_analytic_value(self) -> None:
        discrepancy = LatentStateDiscrepancy(IdentityEncoder(3), norm="L1")
        obs_t = torch.zeros(1, 3)
        obs_tp1 = torch.tensor([[3.0, -4.0, 1.0]])
        self.assertTrue(torch.allclose(discrepancy(obs_t, obs_tp1), torch.tensor([8.0])))

    def test_rejects_unknown_norm_and_shape_mismatch(self) -> None:
        with self.assertRaises(ValueError):
            LatentStateDiscrepancy(IdentityEncoder(2), norm="L3")
        discrepancy = LatentStateDiscrepancy(IdentityEncoder(2))
        with self.assertRaises(ValueError):
            discrepancy(torch.zeros(2, 2), torch.zeros(3, 2))

    def test_works_with_the_real_bigan_encoder(self) -> None:
        encoder = BiGANEncoder((6,), latent_dim=4, feature_dim=8, hidden_dim=8)
        discrepancy = LatentStateDiscrepancy(encoder, latent_dim=4)
        distances = discrepancy(torch.randn(5, 6), torch.randn(5, 6))
        self.assertEqual(distances.shape, (5,))
        self.assertTrue(bool((distances >= 0).all()))

    def test_freeze_disables_encoder_gradients(self) -> None:
        encoder = BiGANEncoder((6,), latent_dim=4, feature_dim=8, hidden_dim=8)
        discrepancy = LatentStateDiscrepancy(encoder)
        discrepancy.freeze()
        self.assertFalse(any(p.requires_grad for p in encoder.parameters()))


class EnsembleRewardVarianceTest(unittest.TestCase):
    """Validate the diversity-enhanced scaling factor ``zeta(r)``."""

    def test_variance_matches_manual_population_variance(self) -> None:
        ensemble = EnsembleRewardVariance(input_dim=3, ensemble_size=4, hidden_dim=8)
        observations = torch.randn(7, 3)
        features = ensemble.features(observations)
        with torch.no_grad():
            predictions = torch.stack([m(features) for m in ensemble.models], dim=-1)
        expected = predictions.var(dim=-1, unbiased=False)
        self.assertTrue(torch.allclose(ensemble.get_variance(observations), expected, atol=1e-6))

    def test_identical_members_produce_zero_variance(self) -> None:
        ensemble = EnsembleRewardVariance(input_dim=3, ensemble_size=3, hidden_dim=8)
        reference = ensemble.models[0].state_dict()
        for model in ensemble.models[1:]:
            model.load_state_dict(reference)
        variance = ensemble.get_variance(torch.randn(4, 3))
        self.assertTrue(torch.allclose(variance, torch.zeros(4), atol=1e-7))

    def test_update_fits_a_constant_reward(self) -> None:
        torch.manual_seed(0)
        ensemble = EnsembleRewardVariance(
            input_dim=3,
            ensemble_size=3,
            hidden_dim=32,
            learning_rate=1e-2,
            batch_size=16,
            min_buffer_size=16,
            buffer_capacity=256,
        )
        observations = torch.randn(64, 3)
        rewards = torch.ones(64)
        first = ensemble.update(observations, rewards)
        self.assertTrue(first.updated)
        for _ in range(200):
            metrics = ensemble.update()
        self.assertLess(metrics.mean_loss, first.mean_loss)
        with torch.no_grad():
            predictions = ensemble.models[0](ensemble.features(observations))
        self.assertLess(float((predictions - 1.0).abs().mean()), 0.2)

    def test_bootstrap_masking_keeps_buffers_distinct(self) -> None:
        torch.manual_seed(0)
        ensemble = EnsembleRewardVariance(
            input_dim=2,
            ensemble_size=4,
            bootstrap_probability=0.5,
            min_buffer_size=8,
            batch_size=8,
        )
        ensemble.add(torch.randn(200, 2), torch.randn(200))
        sizes = {len(buffer) for buffer in ensemble.buffers}
        self.assertGreater(len(sizes), 1)

    def test_rejects_degenerate_ensemble_size(self) -> None:
        with self.assertRaises(ValueError):
            EnsembleRewardVariance(input_dim=2, ensemble_size=1)

    def test_reward_net_rejects_wrong_feature_width(self) -> None:
        with self.assertRaises(ValueError):
            RewardNet(4)(torch.zeros(2, 5))


class ConstantVarianceEnsemble:
    """Ensemble stub returning a fixed variance, for clamp verification."""

    def __init__(self, value: float) -> None:
        self.value = value

    def get_variance(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.full((observations.shape[0],), self.value)


class MetricIntrinsicRewardTest(unittest.TestCase):
    """Validate ``b_t = d_t * min(max(zeta, 1), M)`` and its clamping."""

    def _bonus(self, zeta: float, distance: float = 5.0, maximum: float = 5.0) -> float:
        reward = MetricIntrinsicReward(
            encoder=IdentityEncoder(3),
            ensemble=ConstantVarianceEnsemble(zeta),
            max_reward_scaling=maximum,
            normalize=False,
        )
        obs_t = torch.zeros(1, 3)
        obs_tp1 = torch.tensor([[distance, 0.0, 0.0]])
        return float(reward(obs_t, obs_tp1).combined_score.item())

    def test_variance_below_one_is_clamped_up(self) -> None:
        self.assertAlmostEqual(self._bonus(0.25), 5.0, places=5)

    def test_variance_inside_the_interval_scales_linearly(self) -> None:
        self.assertAlmostEqual(self._bonus(2.0), 10.0, places=5)

    def test_variance_above_the_maximum_is_clamped_down(self) -> None:
        self.assertAlmostEqual(self._bonus(50.0), 25.0, places=5)

    def test_without_an_ensemble_the_bonus_is_the_metric_distance(self) -> None:
        reward = MetricIntrinsicReward(
            encoder=IdentityEncoder(3), ensemble=None, normalize=False
        )
        components = reward(torch.zeros(2, 3), torch.tensor([[3.0, 4.0, 0.0], [0.0, 0.0, 1.0]]))
        self.assertTrue(torch.allclose(components.combined_score, torch.tensor([5.0, 1.0])))
        self.assertTrue(torch.allclose(components.bonus_scale, torch.ones(2)))

    def test_normalization_centres_the_bonus_on_the_extrinsic_scale(self) -> None:
        reward = MetricIntrinsicReward(
            encoder=IdentityEncoder(3), ensemble=None, normalize=True
        )
        obs_t = torch.zeros(64, 3)
        obs_tp1 = torch.randn(64, 3)
        extrinsic = torch.zeros(64)
        components = reward(obs_t, obs_tp1, extrinsic_reward=extrinsic)
        # Eq. (5) with mu(r^e)=0 standardizes the bonus: the batch mean of the
        # normalized score is near zero while raw distances are strictly positive.
        self.assertLess(abs(float(components.normalized_score.mean())), 0.5)
        self.assertTrue(bool((components.combined_score > 0).all()))

    def test_components_expose_generic_novelty_aliases(self) -> None:
        reward = MetricIntrinsicReward(encoder=IdentityEncoder(3), normalize=False)
        components = reward(torch.zeros(3, 3), torch.randn(3, 3))
        self.assertTrue(torch.equal(components.pixel_error, components.latent_distance))
        self.assertTrue(torch.equal(components.feature_error, components.bonus_scale))

    def test_rejects_inverted_clamp_interval(self) -> None:
        with self.assertRaises(ValueError):
            MetricIntrinsicReward(
                encoder=IdentityEncoder(3),
                min_reward_scaling=4.0,
                max_reward_scaling=2.0,
            )


class MetricEMEConfigTest(unittest.TestCase):
    """Validate configuration guards for the new experiment variants."""

    def test_defaults_are_disabled_and_valid(self) -> None:
        config = MetricEMEConfig()
        self.assertFalse(config.enabled)
        self.assertEqual(config.ensemble_size, 5)
        self.assertEqual(config.max_reward_scaling, 5.0)
        self.assertEqual(config.latent_norm, "L2")

    def test_invalid_settings_raise(self) -> None:
        for kwargs in (
            {"latent_norm": "L3"},
            {"ensemble_size": 1},
            {"max_reward_scaling": 0.5, "min_reward_scaling": 1.0},
            {"ensemble_bootstrap_probability": 0.0},
            {"ensemble_input": "pixels"},
        ):
            with self.subTest(**kwargs):
                with self.assertRaises(ValueError):
                    MetricEMEConfig(**kwargs)

    def test_metric_bonus_conflicts_with_transition_novelty(self) -> None:
        from config import NoveltyConfig

        with self.assertRaises(ValueError):
            ExperimentConfig(
                novelty=NoveltyConfig(novelty_type="transition"),
                metric_eme=MetricEMEConfig(enabled=True),
            )


if __name__ == "__main__":
    unittest.main()
