# Adventurer

Exploration with BiGAN for Deep Reinforcement Learning

This repository contains a from-scratch PyTorch implementation of the
Adventurer approach: PPO augmented with BiGAN-based novelty estimation and
intrinsic rewards.

The implementation is designed as a modular research foundation rather than a
minimal example. PPO, GAE, rollout storage, BiGAN components, novelty
estimation, reward shaping, reproducibility utilities, logging, and the
training pipeline are implemented directly in Python and PyTorch.

> **Research note:** Reproducing published benchmark numbers requires matching
> the paper's exact environment versions, preprocessing, network details,
> training budget, evaluation protocol, and hardware/software stack. This
> repository records configurations and separates algorithmic components so
> those details can be audited and modified explicitly.

## Features

- Python 3.9-compatible codebase
- PyTorch 2.x implementation
- PPO implemented from scratch
- Clipped policy objective
- Entropy bonus
- Critic value loss
- Optional value clipping
- GAE and bootstrapped returns
- Gradient-norm clipping
- Minibatch PPO updates
- Linear learning-rate scheduling
- BiGAN encoder, generator, and joint discriminator
- Adversarial BiGAN optimization from a replay buffer
- Pixel reconstruction novelty
- Joint-feature matching novelty
- Running novelty normalization
- Intrinsic reward scaling and clipping
- Explicit extrinsic/intrinsic reward combination
- Gymnasium-compatible environment adapter
- Deterministic seeding utilities
- JSONL and text experiment logging
- Checkpoint-friendly replay and normalization state
- Unit tests based on analytically calculated expected values

## Algorithm Overview

For an observation \(x_t\), the BiGAN encoder produces a latent code:

\[
z_t = E_\psi(x_t).
\]

The generator reconstructs the observation:

\[
\hat{x}_t = G_\theta(E_\psi(x_t)).
\]

Novelty is computed from both pixel and learned feature discrepancies:

\[
e_{pixel}(x_t)
= \frac{1}{d_x}\|x_t - \hat{x}_t\|_2^2,
\]

\[
e_{feature}(x_t)
= \frac{1}{d_h}
\left\|
 h_\omega(x_t,E_\psi(x_t))
 - h_\omega(\hat{x}_t,E_\psi(x_t))
\right\|_2^2,
\]

\[
B(x_t) =
\alpha L_G(x_t) + (1-\alpha)L_D(x_t),
\qquad \alpha\in[0,1].
\]

The paper evaluates a sweep over
\(\alpha\in\{0.5,0.7,0.9,1.0\}\), with \(\alpha=0.9\) as the default
reported setting. The value is configured through `NoveltyConfig.alpha`; the
pixel and feature terms are therefore a convex combination, not an
independent weighted sum.

The normalized novelty score becomes intrinsic reward:

\[
r_t^{int} = \eta\,\widetilde n(x_t).
\]

The reward used by PPO is:

\[
r_t = r_t^{ext} + \rho r_t^{int}.
\]

GAE is then computed from this combined reward:

\[
\delta_t = r_t + \gamma(1-m_t)V_\phi(s_{t+1}) - V_\phi(s_t),
\]

\[
\hat A_t = \delta_t +
\gamma\lambda(1-m_t)\hat A_{t+1},
\]

\[
\hat R_t = V_\phi(s_t) + \hat A_t.
\]

PPO optimizes the clipped policy objective:

\[
r_t(\vartheta) =
\exp\left(
\log\pi_\vartheta(a_t|s_t)
-
\log\pi_{old}(a_t|s_t)
\right),
\]

\[
L^{CLIP}(\vartheta) =
\mathbb E_t\left[
\min\left(
 r_t(\vartheta)\hat A_t,
 \operatorname{clip}(r_t(\vartheta),1-\epsilon,1+\epsilon)\hat A_t
\right)
\right].
\]

The implemented minimization objective is:

\[
L = -L^{CLIP} + c_vL^{VF} - c_eL^{ENT}.
\]

## Data Flow

The environment/PPO/BiGAN integration is explicit:

```text
reset environment
    |
    v
current observation s_t
    |
    +--> BiGAN encoder E(s_t)
    |        |
    |        +--> generator G(E(s_t))
    |        +--> pixel novelty
    |        +--> feature novelty
    |        +--> normalized novelty
    |                 |
    |                 v
    |          intrinsic reward
    |
    +--> actor pi(a_t | s_t)
    +--> critic V(s_t)
             |
             v
       environment.step(a_t)
             |
             +--> extrinsic reward
             +--> terminated/truncated flags
             +--> next observation
                         |
                         v
       combine rewards and store transition
                         |
                         +--> PPO rollout buffer
                         +--> BiGAN replay buffer

rollout complete
    |
    +--> GAE and returns
    +--> PPO minibatch updates
    +--> BiGAN replay updates
    +--> metrics logging
```

## Repository Structure

```text
Adventurer/
├── README.md
├── config.py
├── main.py
├── trainer.py
├── agent/
│   ├── actor.py
│   ├── critic.py
│   ├── ppo.py
│   ├── rollout_buffer.py
│   └── advantage.py
├── bigan/
│   ├── encoder.py
│   ├── generator.py
│   ├── discriminator.py
│   ├── losses.py
│   └── trainer.py
├── exploration/
│   ├── novelty.py
│   ├── transition_novelty.py
│   ├── state_discrepancy.py
│   ├── ensemble_scaling.py
│   ├── episodic_memory.py
│   └── intrinsic_reward.py
├── utils/
│   ├── logger.py
│   ├── normalization.py
│   ├── replay.py
│   └── seed.py
├── environments/
│   └── adapter.py
└── tests/
    └── test_metric_eme.py
```

## Installation

Create a Python 3.9 virtual environment:

```bash
python3.9 -m venv .venv
source .venv/bin/activate
```

Install the core dependencies:

```bash
python -m pip install --upgrade pip
python -m pip install torch numpy gymnasium
```

For Atari environments, install the required Gymnasium extras separately. The
exact package set depends on the selected environment and platform. For
example:

```bash
python -m pip install "gymnasium[atari,accept-rom-license]"
```

Verify the Python version:

```bash
python --version
```

The project targets Python 3.9 and a PyTorch 2.x release that supports Python 3.9.

## Running an Experiment

The default command uses `CartPole-v1`, which is useful for validating the
pipeline interface:

```bash
python main.py \
  --environment-id CartPole-v1 \
  --seed 0 \
  --total-environment-steps 10000 \
  --output-directory runs/cartpole
```

A longer image-based run can be launched with an Atari environment:

```bash
python main.py \
  --environment-id ALE/Breakout-v5 \
  --seed 0 \
  --total-environment-steps 10000000 \
  --rollout-steps 128 \
  --minibatch-size 32 \
  --device auto \
  --output-directory runs/breakout
```

Disable intrinsic rewards for an extrinsic-only PPO ablation:

```bash
python main.py \
  --environment-id CartPole-v1 \
  --disable-intrinsic-reward \
  --output-directory runs/cartpole_extrinsic_only
```

Available command-line options can be inspected with:

```bash
python main.py --help
```

## Configuration

All defaults are defined in `config.py` using frozen dataclasses:

- `SeedConfig`: global and deterministic execution settings.
- `EnvironmentConfig`: observation shape, action dimensions, and preprocessing.
- `PPOConfig`: discounting, GAE, clipping, optimization, and scheduling.
- `BiGANConfig`: latent/feature dimensions and adversarial update schedule.
- `NoveltyConfig`: Eq. (4) convex-mixture parameter `alpha` and intrinsic-reward scale. The default is `alpha=0.9`.
- `TrainingConfig`: total steps, device, logging, and reward mixing.
- `ExperimentConfig`: complete cross-component validation.

The command-line entry point discovers the actual environment observation and
action-space dimensions and combines them with the configured algorithmic
hyperparameters.

## PPO Implementation

`agent/actor.py` implements:

- categorical policies for discrete actions,
- diagonal Gaussian policies for continuous actions,
- action sampling,
- deterministic actions,
- log-probabilities,
- entropy.

`agent/critic.py` implements the state-value function.

`agent/rollout_buffer.py` stores the on-policy trajectory:

```text
observation
next observation
action
reward
terminated
truncated
value estimate
old log-probability
extrinsic reward
intrinsic reward
```

`agent/advantage.py` implements the backward GAE recurrence.

`agent/ppo.py` performs:

- old/new log-probability comparison,
- likelihood-ratio clipping,
- policy loss,
- value loss,
- entropy regularization,
- gradient clipping,
- minibatch epochs,
- learning-rate scheduling.

No external PPO implementation is used.

## BiGAN Implementation

The BiGAN modules implement the joint adversarial game:

- `bigan/encoder.py`: \(E_\psi(x)\).
- `bigan/generator.py`: \(G_\theta(z)\).
- `bigan/discriminator.py`: \(D_\omega(x,z)\).
- `bigan/losses.py`: BCE adversarial losses, reconstruction error, feature
  matching, and optional gradient penalty.
- `bigan/trainer.py`: replay sampling and alternating optimization schedule.

The discriminator receives two types of joint pairs:

```text
real pair:      (x, E(x))
generated pair: (G(z), z)
```

BiGAN observations are stored in `utils/replay.py`, independently from PPO's
on-policy rollout buffer.

## Novelty and Intrinsic Rewards

`exploration/novelty.py` provides independently testable components:

- `PixelReconstructionError`
- `FeatureMatchingError`
- `CombinedNoveltyScore`
- `NormalizedNoveltyScore`
- `NoveltyEstimator`

`exploration/intrinsic_reward.py` provides:

- intrinsic reward scaling,
- optional clipping,
- intrinsic reward enable/disable behavior,
- extrinsic/intrinsic reward combination.

Novelty inference runs without gradients. BiGAN parameters are updated only in
the BiGAN trainer.

## Reproducibility

Use a fixed seed:

```bash
python main.py --environment-id CartPole-v1 --seed 0
```

The seed utility configures:

- Python `random`,
- NumPy,
- PyTorch CPU,
- PyTorch CUDA,
- cuDNN deterministic mode,
- Gymnasium reset/action-space seeds,
- worker-local random streams.

Each run writes its configuration to:

```text
runs/<run-name>/config.json
```

Metrics are written to:

```text
runs/<run-name>/metrics.jsonl
runs/<run-name>/run.log
```

Determinism is hardware, driver, CUDA, PyTorch, and environment dependent.
The recorded seed should therefore be treated as necessary but not always
sufficient for bitwise reproducibility across machines.

## Testing

Run the currently included unit test with:

```bash
python -m unittest discover -s tests -p "test_*.py" -v
```

The advantage tests analytically verify:

- nonterminal bootstrap,
- terminal masking,
- truncation bootstrap,
- multistep GAE,
- return targets,
- normalization statistics.

Additional tests should be added before publication-scale experiments for:

- actor distribution shapes,
- critic outputs,
- rollout storage and minibatches,
- PPO clipping and gradient updates,
- BiGAN tensor shapes and adversarial updates,
- novelty components,
- environment API conversion,
- end-to-end short runs.

## Engineering Conventions

- Mathematical quantities are represented explicitly in code.
- Every module has a module-level docstring.
- Public classes and functions use type hints and docstrings.
- Configuration is immutable after construction.
- PPO and BiGAN sample stores are separated.
- Environment termination and truncation are represented separately.
- Generator/discriminator logits are passed to stable BCE-with-logits losses.
- Novelty statistics are detached from neural-network computation graphs.
- Logging is dependency-light and machine-readable.

## Transition-Level Novelty Variant

The project supports two intrinsic-reward strategies:

```text
novelty_type=state
novelty_type=transition
```

The default `state` strategy preserves the original Adventurer state novelty:

\[
B(s)=\alpha L_G(s)+(1-\alpha)L_D(s).
\]

The transition strategy reuses the BiGAN encoder, generator, and discriminator
and adds a latent forward model:

\[
z_t=E(s_t),\qquad z_{t+1}=E(s_{t+1}),
\]

\[
\hat z_{t+1}=f_\phi(z_t,a_t).
\]

Its transition novelty is:

\[
T(s_t,a_t,s_{t+1})=
\alpha_t\left\|z_{t+1}-\hat z_{t+1}\right\|_1
+(1-\alpha_t)\left\|
 f_D(s_{t+1},z_{t+1})-
 f_D(G(\hat z_{t+1}),\hat z_{t+1})
\right\|_1.
\]

Transition samples are stored independently as:

```text
(s_t, a_t, s_(t+1))
```

The forward model is trained only on latent prediction loss. The feature error
is used as part of the transition novelty score and is logged separately.
The original BiGAN adversarial objective is unchanged.

### Train the state-novelty baseline

```bash
python main.py \\
  --environment-id ALE/Solaris-v5 \\
  --novelty-type state \\
  --seed 0 \\
  --num-parallel-envs 1 \\
  --total-environment-steps 1280 \\
  --output-directory runs/solaris-state
```

### Train the transition-novelty variant

```bash
python main.py \\
  --environment-id ALE/Solaris-v5 \\
  --novelty-type transition \\
  --transition-alpha 0.9 \\
  --seed 0 \\
  --num-parallel-envs 1 \\
  --total-environment-steps 1280 \\
  --output-directory runs/solaris-transition
```

Use identical seeds, environment settings, PPO settings, rollout length, and
training budgets when comparing methods. Only `novelty_type` should differ.

### Transition TensorBoard metrics

Transition runs additionally log:

```text
transition/forward_prediction_loss
transition/feature_matching_loss
transition/total_loss
```

### Generate the comparison figure

After both runs contain `metrics.jsonl` files, generate the comparison artifacts:

```bash
python - <<'PY'
from evaluation.comparison import save_game_score_comparison

csv_path, png_path = save_game_score_comparison(
    "runs/solaris-state/metrics.jsonl",
    "runs/solaris-transition/metrics.jsonl",
    "runs/solaris-comparison",
    smoothing_window=10,
    dpi=300,
)
print(csv_path)
print(png_path)
PY
```

Outputs:

```text
runs/solaris-comparison/game_score_comparison.csv
runs/solaris-comparison/game_score_comparison.png
```

The CSV contains raw and rolling game scores for both variants. The PNG uses
identical axes and a 300 DPI publication-style figure.

## Metric-Based Exploration Bonus (BiGAN Latent + EME Scaling)

Adventurer scores a single state by how badly the BiGAN reconstructs it. This
contribution replaces that reconstruction-error novelty with a *metric* bonus
computed directly in the encoder's latent space and scaled by the epistemic
disagreement of an ensemble of reward models, as in EME:

\[
b_t=\underbrace{\left\|E_\psi(s_t)-E_\psi(s_{t+1})\right\|_p}_{\text{latent state discrepancy}}
\cdot
\underbrace{\min\!\big(\max(\zeta(r),1),\,M\big)}_{\text{diversity-enhanced scaling}},
\qquad
\zeta(r)=\operatorname{Var}\big(\hat r_1(s_{t+1}),\dots,\hat r_K(s_{t+1})\big).
\]

The bonus needs no generator or discriminator pass: one extra encoder forward
per environment step replaces the reconstruction pipeline. The generator and
discriminator remain in the run because the BiGAN adversarial objective still
trains the encoder that defines the metric.

### Modules

| File | Responsibility |
|------|----------------|
| `exploration/state_discrepancy.py` | `LatentStateDiscrepancy`: `d_t = ||E(s_t)-E(s_{t+1})||_p`, `p in {1,2}`, plus encoder freezing. |
| `exploration/ensemble_scaling.py` | `RewardNet`, `FeatureRewardBuffer`, and `EnsembleRewardVariance`: `K` bootstrapped reward regressors and their prediction variance `zeta(r)`. |
| `exploration/intrinsic_reward.py` | `MetricIntrinsicReward`: assembles `b_t`, clamps `zeta` to `[1, M]`, and optionally applies Eq. (5) reward-scale normalization. |
| `config.py` | `MetricEMEConfig`: `enabled`, `ensemble_scaling`, `ensemble_size` (K), `max_reward_scaling` (M), `latent_norm`, and ensemble/optimization settings. |
| `trainer.py` | Stores `(s_t, s_{t+1})` pairs during `collect_rollouts`, computes `b_t` instead of `NoveltyEstimator` novelty when `metric_eme.enabled`, and trains the ensemble once per PPO update. |

`MetricIntrinsicReward` returns a `MetricNoveltyComponents` object rather than a
bare tensor so the trainer, episodic memory, and logger can consume it exactly
like `NoveltyComponents`; its `normalized_score` field is the intrinsic reward
handed to PPO, and `pixel_error` / `feature_error` are compatibility aliases for
the latent distance and the clamped scale.

### Design notes

- **Ensemble inputs.** Members are MLPs. With `ensemble_input=latent` (default)
  they consume `E(s)`, so the cost is independent of the frame resolution;
  `ensemble_input=observation` flattens raw observations instead.
- **Bootstrap diversity.** Each member owns an independent buffer filled with an
  independently masked subset of the collected data
  (`ensemble_bootstrap_probability`, default `0.5`). Sharing one data set makes
  the members converge and collapses `zeta(r)` to zero.
- **Clamping.** `min(max(zeta,1),M)` means the bonus is never smaller than the
  pure metric distance and never inflated by more than `M`; early in training
  `zeta` is typically far below one, so V3 initially behaves like V2.
- **Normalization.** The raw `b_t` is passed through Adventurer's Eq. (5)
  running normalization by default so the intrinsic stream stays on the
  extrinsic-reward scale and `beta` keeps the same meaning across variants.
  Latent distances are much more concentrated than reconstruction errors, so the
  normalized bonus is bounded by `--bonus-normalization-clip` (default `5.0`);
  pass `--disable-bonus-normalization` to feed the raw `b_t` to PPO.
- **Frozen metric.** `--freeze-encoder-after-updates N` stops the BiGAN encoder
  optimizer after `N` PPO updates (`0` freezes before the first rollout), so the
  latent metric, and therefore the scale of `d_t`, becomes stationary. The
  generator and discriminator keep training.

### Experiment variants

| Variant | Command |
|---------|---------|
| **V1 (Baseline)** Adventurer original | `python main.py --env ALE/MontezumaRevenge-v5 --novelty bigan` |
| **V2** Latent discrepancy only | `python main.py --env ALE/MontezumaRevenge-v5 --novelty latent_discrepancy` |
| **V3 (Ours)** Latent + EME scaling | `python main.py --env ALE/MontezumaRevenge-v5 --novelty latent_discrepancy --eme True --ensemble_K 5` |

`--env` is an alias of `--environment-id`, and `--novelty` is an alias layer over
`--novelty-type`: `bigan` and `state` select Adventurer novelty, `transition`
selects the latent forward-model variant, and `latent_discrepancy` enables the
metric bonus. `--eme True` additionally turns on the ensemble scaling factor, so
V2 is the `zeta == 1` ablation of V3. Combining `--novelty transition` with
`--eme True` is rejected by configuration validation because the metric bonus
replaces, rather than augments, the novelty term.

All three variants can be run over several seeds, with comparison figures, via:

```bash
ENVIRONMENT_ID=ALE/MontezumaRevenge-v5 SEEDS="0 1 2" ./run_metric_eme_comparison.sh
```

### Logged metrics

| Tag | Meaning |
|-----|---------|
| `intrinsic/latent_distance` | mean `||E(s_t)-E(s_{t+1})||_p` |
| `intrinsic/ensemble_variance` | mean raw `zeta(r)` before clamping |
| `intrinsic/bonus_scale` | mean `min(max(zeta,1),M)` |
| `intrinsic/bonus` | mean raw bonus `b_t` |
| `intrinsic/encoder_frozen` | `1.0` once the encoder metric is frozen |
| `ensemble/mean_loss`, `ensemble/buffer_size` | reward-ensemble regression diagnostics |
| `novelty/pixel`, `novelty/feature` | Adventurer's `L_G` and `L_D` in V1; the latent distance and bonus scale in V2/V3 |
| `reward/intrinsic`, `reward/total` | reward streams, directly comparable across variants |

Comparing Adventurer's `pixel_loss` and `feature_loss` against the discrepancy
bonus therefore only requires reading `novelty/pixel`, `novelty/feature`,
`intrinsic/latent_distance`, and `reward/intrinsic` from the same JSONL or
TensorBoard stream of each run.

## Extending the Project

### Parallel environments

The rollout buffer and GAE implementation support a time/environment layout, but
the current top-level adapter and trainer use one environment instance. A
parallel collector can be added while reusing the actor, critic, buffer, GAE,
and PPO modules.

### Custom observation preprocessing

The actor, critic, encoder, discriminator, and generator accept vector,
grayscale-image, CHW-image, and HWC-image observations. Environment-specific
frame stacking, cropping, reward clipping, and life-loss handling should be
implemented in an explicit environment wrapper and recorded in the experiment
configuration.

### Research ablations

Useful ablations include:

- extrinsic-only PPO,
- `alpha=1.0` pixel novelty only,
- `alpha=0.0` feature novelty only,
- `alpha` sweep over `{0.5, 0.7, 0.9, 1.0}`,
- intrinsic reward coefficient sweeps,
- BiGAN update frequency sweeps,
- replay-capacity sweeps,
- different latent dimensions,
- different PPO clipping coefficients,
- `latent_norm` in `{L1, L2}` for the metric bonus,
- `max_reward_scaling` (M) and `ensemble_size` (K) sweeps,
- frozen vs. continually trained encoder metric.

## License and Citation

Add the project license and the paper citation before public distribution.
Experiments should cite the Adventurer paper and report the exact configuration
file, environment version, seed, hardware, and evaluation protocol used.
