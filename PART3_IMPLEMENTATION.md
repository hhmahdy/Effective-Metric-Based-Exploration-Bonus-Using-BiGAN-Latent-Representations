# Part-3 Implementation Audit

This document records the implementation of the Part-3 representation-intervention
experiment, following the mandatory corrections.

> **Branch note:** The session harness binds all work to the session branch
> `arena/01a078b9-effective-metric-based-explora`, so a separate `part3-intervention`
> git branch was **not** created. All of the isolation that the requested branch
> would provide is instead preserved by an explicit `--part3` mode plus **untouched
> legacy code paths** (legacy `--novelty transition` and `--novelty bigan` behave
> exactly as before, verified below).

---

## A. Files modified

| File | Change |
|------|--------|
| `config.py` | Added `Part3Config` dataclass; added `part3` field to `ExperimentConfig`; added Part-3 cross-validation (reject `part3` + `metric_eme`; require `novelty_type == "state"` under `--part3`). |
| `main.py` | Added `--part3`, `--representation {bigan,idf,rnd}`; made `--total-environment-steps` default to `None` and resolve to the Stage-1 exact budget under `--part3`; wired `Part3Config` into `build_config`; dispatches to `Part3Trainer` when `--part3`. |
| `README.md` | Documented the Part-3 mechanism, representations, causal test, interpretation rules, RNG isolation, Stage-1 budget, and the CorridorTV blocker. |
| `.gitignore` | Added Python/venv/experiment-artifact ignores. |

No legacy implementation file was modified: `exploration/transition_novelty.py`,
`exploration/novelty.py`, `trainer.py`, `bigan/*`, `agent/*`, and `environments/*`
are untouched.

## B. Files added

| File | Purpose |
|------|---------|
| `part3/__init__.py` | Package marker. |
| `part3/rng.py` | Named dedicated RNG streams and `make_generator` / `run_under_rng` primitives. |
| `part3/representations.py` | `BiGANRepresentation`, `IDFRepresentation`, `RNDRepresentation`, the `build_representation` factory, and `describe_representation`. |
| `part3/transition_novelty.py` | Pure `N_T` estimator (`Part3TransitionNoveltyEstimator`) and the shared MSE forward-model trainer (`Part3ForwardModelTrainer`). |
| `part3/trainer.py` | `Part3Trainer`: the vectorized Part-3 causal training loop. |
| `tests/test_part3.py` | 24 unit + smoke tests. |
| `PART3_IMPLEMENTATION.md` | This document. |

## C. Exact Part-3 transition novelty

```
N_T(s_t, a_t, s_{t+1}) = || f(E(s_t), a_t) - E(s_{t+1}) ||_2
```

- `E` is the representation under test (`bigan`, `idf`, or `rnd`).
- `f` is one shared `LatentForwardModel` (identical architecture for all three).
- The norm is the **L2 norm** over the latent vector: `sqrt(sum((f - E(s'))^2))`.

The Part-3 transition path contains **no** BiGAN discriminator features, no
generator reconstruction, no feature matching, no EME, no kNN, no visit counts,
no episodic counts, no ensemble uncertainty, no adaptive scaling, no state
novelty, and no death/respawn bonus. Death/respawn transitions are neither
masked nor rewarded; they are logged by the trainer.

Implementation: `part3/transition_novelty.py::Part3TransitionNoveltyEstimator.score`.

## D. Exact forward-model training loss

```
loss = mean( || f(E(s_t), a_t) - E(s_{t+1}) ||_2^2 )
```

This is the squared-L2 / MSE regression. It is **identical** across BiGAN, IDF,
and RND. Only `f` is updated; `E` is used as a frozen target (no gradients flow
to `E` from the forward-model regression). There is **no L1** term anywhere in
the Part-3 transition path.

Implementation: `part3/transition_novelty.py::Part3ForwardModelTrainer.update`.

## E. Representation architectures

All three representations reuse the **same encoder trunk and latent dimension**
(`BiGANEncoder` architecture: conv/MLP feature trunk -> `feature_dim` -> LeakyReLU
-> `latent_head` -> `latent_dim`). The latent dimension is `128` by default for all
three. The objectives differ, but the encoder capacity is identical, which is what
makes the causal test clean.

Parameter counts for the default Stage-1 image observation shape `(84, 84, 4)`,
`latent_dim=128`, `feature_dim=256`, `hidden_dim=512`:

| Arm | Encoder params | Extra modules |
|-----|---------------|---------------|
| BiGAN | 3,687,232 | generator 437,140; discriminator 4,081,601 |
| IDF | 3,687,232 | inverse-dynamics head 132,612 |
| RND | 3,687,232 | frozen random target 3,687,232 (not trained) |

Shared forward model `LatentForwardModel(latent=128, action=4, hidden=256)`: **132,736** params.

| Aspect | BiGAN | IDF | RND |
|--------|-------|-----|-----|
| Objective | adversarial generative (BiGAN joint distribution matching) | inverse-dynamics / action-predictive (`a_t` from `E(s_t),E(s_{t+1})`) | random-target prediction (match frozen `T(s)`) |
| Optimizer | Adam (encoder/generator/discriminator) | Adam (encoder + head) | Adam (predictor only; target frozen) |
| Learning rate | 2e-4 | 2e-4 | 2e-4 |
| Batch size | 64 | 64 | 64 |
| Update frequency | 1 per PPO update | 1 per PPO update | 1 per PPO update |
| Initialization | orthogonal (dedicated RNG stream) | orthogonal (dedicated RNG stream) | orthogonal; target under dedicated RND stream |
| Training data | observation replay | transition replay | observation replay |
| Updates | online (counted at runtime) | online (counted at runtime) | online (counted at runtime) |
| Online / frozen | online | online | predictor online; target frozen |

No per-representation learning-rate, reward-coefficient, PPO, forward-model,
normalization, or training-budget tuning is performed.

## F. RNG isolation design

Dedicated streams are defined in `part3/rng.py`, each derived from the experiment
seed via `derive_seed`, so they are mutually isolated but move together with the
seed:

| Stream | Constant | Used for |
|--------|----------|----------|
| PPO init | `RNG_PPO_INIT` | actor/critic initialization |
| Actor/critic init | `RNG_ACTOR_CRITIC_INIT` | (reserved; folded into PPO init) |
| Representation init | `RNG_REPRESENTATION_INIT` | encoder / head / generator / discriminator init |
| Representation training | `RNG_REPRESENTATION_TRAIN` | IDF/RND training (replay sampling) |
| Forward-model init | `RNG_FORWARD_MODEL_INIT` | shared `LatentForwardModel` init |
| Forward training | `RNG_FORWARD_TRAIN` | forward-model optimizer / replay sampling |
| Replay sampling | `RNG_REPLAY_SAMPLING` | replay-buffer minibatch indices |
| BiGAN latent sampling | `RNG_BIGAN_LATENT` | BiGAN adversarial update |
| RND target init | `RNG_RND_TARGET` | RND frozen target initialization |

Network construction is scoped with `run_under_rng`, which forks the global RNG,
sets it to the stream's state, and restores it afterward. Replay sampling uses
explicit `torch.Generator` objects where the buffer API accepts them, and the
legacy BiGAN trainer (which draws replay/latent samples from the implicit global
RNG) is run entirely inside a forked RNG scope.

This does **not** claim trajectories remain bit-identical across representations.
The stated requirement is:

> "Same seed and controlled RNG initialization before behavioral divergence."

Once the different representations produce different intrinsic rewards, their
trajectories are expected to diverge; that divergence is the experimental effect.

## G. Configuration example

```bash
python main.py \
  --environment-id ALE/MontezumaRevenge-v5 \
  --part3 --representation bigan \
  --seed 0 --num-parallel-envs 96 --rollout-steps 128 \
  --minibatch-size 32 --device auto \
  --output-directory runs/part3/bigan
```

`--representation` = `bigan` | `idf` | `rnd`. With `--part3` and no explicit
`--total-environment-steps`, the Stage-1 budget is used:

- 96 environments × 128 rollout steps = **12,288** env-steps per update.
- The requested `~2M` budget is not divisible by 12,288, so the exact run uses
  **1,990,656** env-steps = **162 updates**, documented as "approximately 2M
  steps; exact budget constrained by rollout divisibility."

Legacy commands (e.g. `--novelty transition`, `--novelty bigan`,
`--novelty latent_discrepancy`) are unchanged and leave `--part3` off.

## H. Test results

```
$ python -m unittest discover -s tests -p "test_*.py"
...
Ran 72 tests in 2.614s
OK
```

- **48** pre-existing tests: pass (backward compatibility preserved).
- **24** new Part-3 tests (`tests/test_part3.py`): pass.

The new tests cover: (1) L2 novelty mathematical correctness (analytic L2 and
"not L1"), (2) MSE forward-model loss (analytic squared-L2), (3) action one-hot
encoding, (4) BiGAN representation shape, (5) IDF representation shape,
(6) RND representation shape, (7) identical forward-model architecture across
representations, (8) configuration validation, (9) dedicated RNG streams,
(10) PyTorch state round-trip (checkpoint-ability), and (11) one tiny CPU smoke
run for each representation.

Note on checkpointing: Part 3 does **not** implement a trainer-level checkpoint as
a requirement for running the experiment. Explicit checkpointing is for
reproducibility/debugging only. The pre-requisite -- that the representation
encoder and shared forward model can be snapshotted and restored exactly -- is
verified by the `StateRoundTripTest`.

## I. CPU smoke-test results

One small CPU run per representation (`CartPole-v1`, 2 parallel envs, 4 rollout
steps, `batch_size=4`, 8 updates) verified the full pipeline
`environment -> E(s) -> f(E(s),a) -> L2 N_T -> intrinsic reward -> PPO` without
shape/runtime errors:

| Representation | updates | rep loss | rep updated | forward MSE | forward updated | mean N_T | death count |
|----------------|---------|----------|-------------|-------------|-----------------|----------|-------------|
| bigan | 8 | 1.4229 | yes | 0.1170 | yes | 0.3311 | 1 |
| idf | 8 | 0.6990 | yes | 0.1192 | yes | 0.3384 | 0 |
| rnd | 8 | 0.0001 | yes | 0.0998 | yes | 0.3138 | 1 |

The same three arms also pass as unit tests in `TinyCPUSmokeTest`, and the
end-to-end CLI (`main.py --part3 --representation bigan ...`) exits 0 and writes
`metrics.jsonl` with `novelty/NT`, `forward_model/mse`, `representation/loss`,
and `death/*` tags.

## J. Remaining blockers

> Updated: `CorridorTV` is now implemented and registered (see
> `environments/corridortv.py` and `tests/test_corridortv.py`), so the Stage-1
> environment blocker is **resolved**. Stage-1 and Stage-2 launchers are
> implemented in `run_part3_stage1.sh` and `run_part3_stage2.sh`.

1. **Separate `part3-intervention` branch was not created** because the session
   harness binds this work to `arena/01a078b9-effective-metric-based-explora`.
   All isolation is preserved via the explicit `--part3` mode and untouched
   legacy code paths.

2. Trainer-level checkpoint serialization is intentionally **not** required to
   run the experiment (implemented only for reproducibility/debugging). This is
   the only one of the requested tests that is not a full round-trip of a
   trainer-level checkpoint.

3. Stage 1 and Stage 2 have **not** been launched. Stage 2 requires
   `gymnasium[atari]` / `ale_py` (and the ALE environment) to resolve
   `ALE/MontezumaRevenge-v5`; in this sandbox `ale_py` is not installed, so the
   Stage-2 registration preflight exits with the resolution error until that
   package is available.
