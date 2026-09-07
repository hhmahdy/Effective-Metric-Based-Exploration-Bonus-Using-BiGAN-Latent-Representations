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
├── part3/
│   ├── __init__.py
│   ├── rng.py
│   ├── representations.py
│   ├── transition_novelty.py
│   └── trainer.py
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
- **Scaling mode.** `--eme-mode clamped` (default) is EME as published,
  `min(max(zeta,1),M)`: the bonus is never smaller than the pure metric
  distance and never inflated by more than `M`. That formulation assumes `zeta`
  is of order one, which holds for shaped rewards but not for sparse Atari
  rewards -- with almost all regression targets equal to zero the members agree,
  `zeta << 1`, the lower clamp binds everywhere, and V3 collapses onto V2.
  `--eme-mode normalised` divides by the running mean instead,
  `min(zeta/E[zeta], M)`, which is scale-free and therefore ranks transitions
  correctly at any reward magnitude. The ratio is exact: adding an epsilon to
  the numerator would pull the factor back towards one in exactly the tiny-`zeta`
  regime the mode exists to rescue. If the running mean itself falls below
  `zeta_epsilon` the ensemble is treated as degenerate and the factor falls back
  to one, so the bonus degrades to V2 rather than to zero reward.
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
| **V3** Latent + clamped `zeta` (EME exactly) | `python main.py --env ALE/MontezumaRevenge-v5 --novelty latent_discrepancy --eme True --ensemble_K 5` |
| **V4 (Ours)** Latent + normalised `zeta` | `python main.py --env ALE/MontezumaRevenge-v5 --novelty latent_discrepancy --eme True --eme-mode normalised --ensemble_K 5` |

The four variants isolate one factor each: V2 removes EME's scaling from the
metric bonus, V3 restores it in its published clamped form, and V4 replaces the
clamp with mean normalization. On a sparse-reward game V3 and V2 are expected to
coincide wherever `zeta < 1`, which is what makes V4 the informative comparison
rather than a redundant fourth curve.

A 512-step CartPole run makes the collapse concrete. Over sixteen PPO updates
the raw ensemble variance rises from `4.6e-5` to `7.5e-3` and never approaches
one, so the two modes see identical `zeta` and produce very different factors:

| Update | `zeta` | V3 scale (clamped) | V4 scale (normalised) |
|--------|--------|--------------------|-----------------------|
| 1 | 4.6e-5 | 1.000 | 2.68 |
| 6 | 2.8e-4 | 1.000 | 3.41 |
| 11 | 1.6e-3 | 1.000 | 3.40 |
| 16 | 5.3e-3 | 1.000 | 2.74 |

V3's factor is constant at the clamp floor for every update, so its bonus is
bit-for-bit V2's. V4's factor varies with the ensemble and, because `E[zeta]` is
an EMA rather than a cumulative mean, it tracks the rising variance instead of
saturating at `M`. `--zeta-momentum` (default `0.99`, roughly a hundred-batch
horizon) sets how quickly the reference level adapts.

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
| `intrinsic/ensemble_variance` | mean raw `zeta(r)` before scaling |
| `intrinsic/mean_ensemble_variance` | running `E[zeta]`, the normalised mode's divisor |
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
- `eme_mode` in `{clamped, normalised}`,
- frozen vs. continually trained encoder metric.

## Part 3: Representation-Intervention Causal Test

Part 3 is a **causal attribution** study, not an algorithm-development stage.
It holds every component of the system fixed and varies only the observation
representation `E` that feeds the transition novelty.

### Mechanism

The Part-3 transition novelty is defined exactly as:

```
N_T(s_t, a_t, s_{t+1}) = || f(E(s_t), a_t) - E(s_{t+1}) ||_2
```

where `E` is the representation under test and `f` is a **single shared**
`LatentForwardModel` used by all three arms.

- **Metric** = L2 norm, `||·||_2`, over the latent vector.
- **Forward-model training loss** = squared L2 / MSE:

```
loss = mean( || f(E(s_t), a_t) - E(s_{t+1}) ||_2^2 )
```

This loss is identical across the three representations. **No L1 term appears
anywhere in the Part-3 transition path.**

The Part-3 path deliberately contains none of the following: BiGAN
discriminator features, generator reconstruction, feature matching, EME, kNN,
visit counts, episodic counts, ensemble uncertainty, adaptive scaling, state
novelty, or any death/respawn bonus. Death/respawn transitions are not masked
from `N_T` and receive no bonus; the trainer only logs `death/count`,
`death/fraction_high`, and `death/novelty_percentile`.

The raw `N_T` is converted to an intrinsic reward by the same Eq. (5) reward
processing used by the rest of the pipeline (running normalization onto the
extrinsic-reward scale). This is the **shared reward processing** across the
three arms and is what keeps the intrinsic reward comparable even though the
three representations produce latents on different scales.

### The three representations

| Arm | `E` | Objective |
|-----|-----|-----------|
| **BiGAN** | BiGAN encoder | **adversarial generative** representation: `(x, E(x))` and `(G(z), z)` joint distributions made indistinguishable |
| **IDF** | encoder + inverse-dynamics head | **inverse-dynamics / action-predictive** representation: `a_t` predicted from `(E(s_t), E(s_{t+1}))` |
| **RND** | predictor network (frozen random target) | **random-target prediction** representation: `E(s) ~= T(s)` for a frozen random `T` |

The three representations are **not** interchangeable encoders; their
objectives define genuinely different representation-learning problems. They
all reuse the same encoder trunk and latent dimension, use the same batch size,
update frequency, training-start condition, and replay capacity, and are all
trained **online** during RL (no pretraining/freeze schedule, no checkpoints
required to run).

There is **no per-representation hyperparameter tuning**: no separate
learning-rate, reward-coefficient, PPO, forward-model, normalization, or
training-budget tuning. `E` is the only intervention.

### Primary causal test

```
E_BiGAN -> N_T -> PPO
E_IDF   -> N_T -> PPO
E_RND   -> N_T -> PPO
```

with identical environment, PPO, `N_T`, forward model, reward processing,
budget, and seeds; only `E` differs.

### Interpretation rule

Do **not** assume IDF will win. Possible outcomes:

- **A.** IDF > BiGAN → supports the representation-bottleneck hypothesis.
- **B.** BiGAN ≈ IDF → weak/no evidence for a representation bottleneck.
- **C.** RND/IDF/BiGAN all perform poorly → suspect the transition-novelty
  mechanism rather than the representation.
- **D.** BiGAN > IDF despite weaker action prediction → important result;
  investigate why.
- **E.** High seed variance → report instability; do not hide it.
- **F.** Better forward prediction does not improve exploration → prediction
  quality alone is insufficient.

### Configuration example

```bash
python main.py \
  --environment-id ALE/MontezumaRevenge-v5 \
  --part3 --representation bigan \
  --seed 0 --num-parallel-envs 96 --rollout-steps 128 \
  --minibatch-size 32 --device auto \
  --output-directory runs/part3/bigan
```

`--representation` chooses `bigan`, `idf`, or `rnd`. The Stage-1 budget is
`96 * 128 = 12,288` environment steps per update. The requested `~2M` budget is
not divisible by `12,288`, so the exact run uses **`1,990,656`** environment
steps = **162 updates**. This is documented as:

> approximately 2M steps; exact budget constrained by rollout divisibility.

`--total-environment-steps` defaults to the Stage-1 budget when `--part3` is
set (and to the legacy `12,288,000` otherwise).

### RNG isolation

Part 3 uses dedicated RNG streams for PPO/actor/critic initialization,
representation initialization, representation training, forward-model
initialization, forward-model training, replay sampling, BiGAN latent sampling,
and RND target initialization (see `part3/rng.py`). This does **not** claim the
trajectories remain bit-identical across representations. The requirement is:

> "Same seed and controlled RNG initialization before behavioral divergence."

Once the different representations produce different intrinsic rewards, their
trajectories are expected to diverge. That divergence is the experimental
effect, not a confounder.

### CorridorTV

The Part-3 Stage-1 environment is implemented from scratch as a PyColab-style
gridworld in `environments/corridortv.py` and registered as `CorridorTV-v0`.
It reproduces the exact observation / reward / info contract called for by the
Part-3 protocol:

- Internal `32 x 32 x 3` grid, exposed as a `(64, 64, 3)` `uint8` observation
  (nearest-neighbor upsampled 32x32 -> 64x64; a **fixed** preprocessing step
  identical across representations and seeds, recorded in `metadata`).
- `Discrete(4)` deterministic movement, a switch tile that toggles a door, and a
  sparse `+1` goal reward (episode terminates).
- Exactly **73** reachable controllable configurations (`agent position` x
  `door state`), validated by exhaustive BFS; `CORRIDORTV_CONTROLLABLE_STATES = 73`.
- Action- and agent-independent noisy TV / flicker region and a random-walk
  colored decoy, both driven by a per-episode seeded RNG that never affects the
  controllable transition.
- `info` contract with `agent_position`, `door_open`, `controllable_state`,
  `goal_reached`, `flicker_active`, `decoy_position`, and `seed`.

See `tests/test_corridortv.py` for the CPU-runnable contract tests.

### Stage launchers

`run_part3_stage1.sh` (CorridorTV) and `run_part3_stage2.sh`
(`ALE/MontezumaRevenge-v5`) launch the Part-3 representation-intervention runs
over seeds x representations. Both scripts:

- run preflight checks (Python imports, budget divisibility, environment
  registration) and create the `results/part3/{raw,processed,figures,checkpoints,logs}`
  tree;
- skip a seed/arm whose run directory already contains `metrics.jsonl` (unless
  `FORCE_RERUN=1`, in which case the old directory is moved to `.prev.$(date +%s)`
  and rerun fresh);
- execute the identical Part-3 command for every arm, varying only
  `--representation`, and tee console output to
  `results/part3/logs/stage{N}/{seed}_{rep}.console.log`;
- collect per-arm exit codes, print an OK/FAILED/SKIPPED manifest, and exit
  non-zero if any arm failed.

Available overrides (environment variables): `ENVIRONMENT_ID`, `SEEDS`,
`REPRESENTATIONS`, `NUM_PARALLEL_ENVS`, `ROLLOUT_STEPS`, `MINIBATCH_SIZE`,
`TOTAL_ENVIRONMENT_STEPS`, `DEVICE`, `PYTHON_BIN`, `OUTPUT_ROOT`, `FORCE_RERUN`.

Stage 1 defaults to `ENVIRONMENT_ID=CorridorTV-v0` with the exact budget
`TOTAL_ENVIRONMENT_STEPS=1990656`; Stage 2 defaults to
`ENVIRONMENT_ID=ALE/MontezumaRevenge-v5` with `TOTAL_ENVIRONMENT_STEPS=12288000`.

The Stage-1 budget is documented as:

> approximately 2M steps; exact budget of 1990656 = 162 updates * 12288
> constrained by rollout divisibility

(2,000,000 is not divisible by 96*128 = 12,288; the 96x128 configuration is
intentionally not changed.)

These scripts only launch runs. Analysis of each run's `metrics.jsonl` (figures,
first-discovery, IQM, diagnostics) is a separate step; the trainer writes its
own periodic snapshots into each run directory and the launcher leaves them
untouched (no resume logic).

## License and Citation

Add the project license and the paper citation before public distribution.
Experiments should cite the Adventurer paper and report the exact configuration
file, environment version, seed, hardware, and evaluation protocol used.
