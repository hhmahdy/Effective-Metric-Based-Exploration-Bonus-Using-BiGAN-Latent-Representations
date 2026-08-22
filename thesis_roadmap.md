# Thesis Roadmap

## Effective Metric-Based Exploration Bonus Using BiGAN Latent Representations

---

## 1. Overview of the Research

This thesis addresses the problem of efficient exploration in deep reinforcement learning (RL), particularly in environments with high-dimensional observations and sparse extrinsic rewards. The work extends the **Adventurer** algorithm (Liu & Liu, 2025)—which uses Bidirectional Generative Adversarial Networks (BiGANs) to estimate state novelty—by replacing its reconstruction-error-based exploration bonus with a **metric-based exploration bonus** computed directly in the BiGAN encoder's latent space and scaled by the epistemic disagreement of an ensemble of reward models, following the **Effective Metric-based Exploration-bonus (EME)** framework (Wang et al., 2024). A key contribution is identifying and addressing a critical limitation of the published EME scaling factor—its collapse under sparse Atari rewards—and proposing a **normalised scaling mode** that remains informative at any reward magnitude.

### Core Contributions (from the codebase and README)

1. **Latent State Discrepancy** (`exploration/state_discrepancy.py`): Replaces Adventurer's reconstruction novelty $B(s) = \alpha L_G(s) + (1-\alpha)L_D(s)$ with a transition-level metric $d_t = \|E_\psi(s_t) - E_\psi(s_{t+1})\|_p$ in the BiGAN encoder's latent space.

2. **Diversity-Enhanced Scaling Factor** (`exploration/ensemble_scaling.py`): Integrates EME's ensemble reward variance $\zeta(r) = \mathrm{Var}(\hat{r}_1(s), \ldots, \hat{r}_K(s))$ to amplify exploration in reward-relevant but poorly modelled regions.

3. **Normalised EME Mode** (`exploration/intrinsic_reward.py`): Replaces EME's published clamped scaling $\min(\max(\zeta, 1), M)$ with a scale-free normalised scaling $\min(\zeta / \mathbb{E}[\zeta], M)$, which uses an exponential moving average (EMA) as the reference level and degrades gracefully to the pure metric bonus when the ensemble is degenerate.

4. **Four Experiment Variants**:
   - **V1** (Baseline): Adventurer original reconstruction novelty $B(s)$
   - **V2**: Latent discrepancy only ($\zeta \equiv 1$)
   - **V3**: Latent discrepancy + clamped EME scaling (published EME)
   - **V4** (Ours): Latent discrepancy + normalised EME scaling

---

## 2. Proposed Chapters and Sections

### Chapter 1: Introduction

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 1.1 Motivation | Establish the exploration problem in deep RL; sparse rewards in high-dimensional spaces | Why is exploration a bottleneck in deep RL? |
| 1.2 Problem Statement | Define the intrinsic reward exploration framework; state the specific problem of metric-based bonuses under sparse rewards | How can we construct an exploration bonus that is both metric-grounded and reward-sensitive? |
| 1.3 Contributions | List the four contributions above | — |
| 1.4 Thesis Structure | Outline subsequent chapters | — |

**Key Papers**: Liu & Liu (2025) [Adventurer]; Wang et al. (2024) [EME]; Burda et al. (2019) [RND]

**Missing Information**: None — the contributions are explicitly defined in the codebase.

---

### Chapter 2: Background and Related Work

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 2.1 Markov Decision Processes and Policy Optimization | Formal MDP framework, value functions, policy gradients | — |
| 2.2 Proximal Policy Optimization (PPO) | PPO clipped surrogate objective, GAE, two-stream advantage | How does PPO balance stable policy updates with exploration? |
| 2.3 Generative Adversarial Networks and BiGANs | Standard GANs, BiGAN joint adversarial game, encoder–generator inversion property | Why does BiGAN learn an invertible mapping, and how does the encoder define a latent metric? |
| 2.4 Intrinsic Reward and Exploration Bonuses | Taxonomy: count-based, prediction-based, uncertainty-based, memory-based | What are the failure modes of existing exploration methods in high-dimensional sparse-reward settings? |
| 2.5 Novelty-Based Exploration Methods | RND, ICM, GAEX, VAE-based methods; their relationship to reconstruction error | How does reconstruction novelty relate to visitation counts? |
| 2.6 Metric-Based Exploration and EME | Bisimulation metric, latent-space state discrepancy, EME's ensemble scaling factor | What are the limitations of current metric-based exploration bonuses, and how does EME address them? |
| 2.7 Episodic Memory and the Resettable Premise | Top-K episodic memory, NGU-style episodic+lifelong novelty, Agent57 | How does the resettable premise prevent intrinsic reward vanishing? |

**Key Papers**:
- Schulman et al. (2017) [PPO]
- Schulman et al. (2015) [GAE/TRPO]
- Donahue et al. (2017) [BiGAN]
- Dumoulin et al. (2017) [ALI]
- Burda et al. (2019) [RND]
- Pathak et al. (2017) [ICM]
- Hong et al. (2019) [GAEX]
- Wang et al. (2024) [EME]
- Badia et al. (2020a) [NGU]
- Badia et al. (2020b) [Agent57]
- Liu & Liu (2025) [Adventurer]
- Bellemare et al. (2016) [Unifying Count-Based]
- Ostrovski et al. (2017) [Count-Based with PixelCNN]

**Missing Information**: The full EME paper (Wang et al., 2024) should be consulted for its theoretical analysis of the bisimulation metric approximation and the formal derivation of the diversity-enhanced scaling factor. The Adventurer paper's Appendix D contains architecture details needed for the BiGAN background.

---

### Chapter 3: The Adventurer Framework

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 3.1 BiGAN-Based State Novelty Estimation | Encoder $E_\psi$, generator $G_\theta$, discriminator $D_\omega$; joint adversarial objective | How does the BiGAN adversarial game induce a novelty estimator? |
| 3.2 Combined Novelty Score | $B(s) = \alpha L_G(s) + (1-\alpha)L_D(s)$; pixel reconstruction + feature matching | Why combine pixel-level and feature-level errors? What is the effect of $\alpha$? |
| 3.3 Intrinsic Reward Normalization | Eq. (5): reward-scale normalization $\tilde{n}(s) = (B(s) - \mu_B + \mu_{r^e}) / \sigma_B$ | How does normalization align intrinsic and extrinsic reward scales? |
| 3.4 Two-Stream PPO Integration | Separate extrinsic/intrinsic GAE, combined advantage $A_t = A_t^e + \beta A_t^i$ | Why separate reward streams rather than mixing rewards before GAE? |
| 3.5 Episodic Memory and Resettable Exploration | Algorithm 2: top-K novel state memory, state restoration | How does the resettable premise double exploration performance? |
| 3.6 Transition Novelty Variant | Latent forward model $f_\phi$, transition novelty $T(s_t, a_t, s_{t+1})$ | Can transition-level novelty improve over state-level novelty? |
| 3.7 Implementation Architecture | Network architectures, training schedule, replay buffers | — |

**)**

**Key Papers**: Liu & Liu (2025) [Adventurer]; Donahue et al. (2017) [BiGAN]; Burda et al. (2019) [RND — two value heads]; Schulman et al. (2017) [PPO]

**Missing Information**: The Adventurer paper's CIFAR-10 validation experiment (Section 5.1.1) provides empirical evidence that BiGAN novelty scores correlate with state novelty; this should be referenced if the full paper is available.

---

### Chapter 4: Metric-Based Exploration Bonus with BiGAN Latent Representations

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 4.1 Motivation and Problem Analysis | Reconstruction novelty is expensive (requires G and D passes); not transition-aware; does not distinguish reward-relevant from reward-irrelevant novelty | What are the computational and conceptual limitations of Adventurer's reconstruction novelty? |
| 4.2 Latent State Discrepancy | $d_t = \|E_\psi(s_t) - E_\psi(s_{t+1})\|_p$; $p \in \{1, 2\}$; one encoder forward per step vs. full reconstruction pipeline | How does the latent metric capture transition novelty without reconstruction? |
| 4.3 EME Diversity-Enhanced Scaling | Ensemble of $K$ bootstrapped reward models; $\zeta(r) = \mathrm{Var}_k(\hat{r}_k(s_{t+1}))$; scaling factor $\min(\max(\zeta, 1), M)$ | How does the ensemble variance identify reward-relevant but poorly modelled regions? |
| 4.4 The Full Exploration Bonus | $b_t = d_t \cdot \min(\max(\zeta(r), 1), M)$; integration with Adventurer's Eq. (5) normalization | How do the metric term and scaling factor interact? |
| 4.5 Implementation Details | `LatentStateDiscrepancy`, `EnsembleRewardVariance`, `MetricIntrinsicReward`; bootstrap diversity; frozen encoder option | — |

) and ensemble scaling (Section 4.3) into the full bonus (Section 4.4).)

**Key Papers**: Wang et al. (2024) [EME]; Liu & Liu (2025) [Adventurer]; Donahue et al. (2017) [BiGAN — encoder as feature representation]

**Missing Information**: A formal theoretical analysis proving that the latent discrepancy $d_t$ is a valid metric on state space (given sufficient encoder capacity) would strengthen this chapter. This is **not** present in the codebase and would need to be derived or cited from EME's theoretical framework.

---

### Chapter 5: Normalised EME Scaling for Sparse Rewards

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 5.1 The Sparse-Reward Collapse Problem | In sparse Atari rewards, almost all regression targets are zero, so ensemble members agree, $\zeta \ll 1$, the lower clamp binds, and V3 collapses to V2 | Why does the published EME scaling factor fail in sparse-reward settings? |
| 5.2 Normalised Scaling Mode | $\mathrm{scale} = \min(\zeta / \mathbb{E}[\zeta], M)$; EMA reference level $\mathbb{E}[\zeta]$ with momentum; scale-free property | How does normalisation restore the scaling factor's discriminative power at any reward magnitude? |
| 5.3 Degeneracy Handling | When $\mathbb{E}[\zeta] < \zeta_\epsilon$, the ensemble carries no usable signal; fall back to scale = 1 (pure metric bonus) | How does the normalised mode degrade gracefully rather than zeroing exploration? |
| 5.4 Reference Level as Exponential Moving Average | $\mathbb{E}[\zeta] \leftarrow m \cdot \mathbb{E}[\zeta] + (1-m) \cdot \mathrm{mean}(\zeta_\mathrm{batch})$; seeded by first batch; why EMA over cumulative mean | Why use an EMA rather than a cumulative mean for the reference level? |
| 5.5 Empirical Demonstration of the Collapse | CartPole 512-step data: raw $\zeta$ rises from 4.6e-5 to 7.5e-3; V3 scale is constant at 1.0; V4 scale varies (2.68 → 3.41 → 3.40 → 2.74) | What does the empirical data show about V3 vs. V4? |
| 5.6 Frozen Encoder for Stationary Metrics | `--freeze-encoder-after-updates N`; BiGAN encoder stops receiving gradients; generator and discriminator continue | Why freeze the encoder to stabilise the latent metric? |

**Key Papers**: Wang et al. (2024) [EME — clamped scaling]; Liu & Liu (2025) [Adventurer — Eq. (5) normalization]

**Missing Information**:
- A **formal proof** that $\zeta / \mathbb{E}[\zeta]$ is scale-free (i.e., invariant under rescaling of the reward) would strengthen Section 5.2. The codebase demonstrates this empirically but does not prove it.
- **Large-scale experimental results** comparing V3 and V4 on hard Atari games (Montezuma's Revenge, Gravitar, Solaris) are described in the README but **not included in the repository**. These results are critical for the thesis and must be generated.
- A **theoretical analysis** of the EMA reference level's convergence properties would strengthen Section 5.4.

---

### Chapter 6: Experimental Evaluation

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 6.1 Experimental Setup | Environments, hyperparameters, training budget, evaluation protocol | — |
| 6.2 BiGAN Validation | CIFAR-10 novelty estimation (from Adventurer paper); Montezuma's Revenge novelty ranking | Does BiGAN accurately estimate state novelty? |
| 6.3 V1–V4 Variant Comparison on Sparse-Reward Atari | Montezuma's Revenge, Gravitar, Solaris; game score, intrinsic reward, total reward | Does V4 outperform V3 on sparse-reward games? Is V3 ≈ V2 where $\zeta < 1$? |
| 6.4 Ablation Studies | $\alpha$ sweep; $\beta$ sweep; ensemble size $K$; max reward scaling $M$; latent norm ($L_1$ vs. $L_2$); frozen vs. continually trained encoder; normalised vs. clamped mode | Which factors contribute most to V4's performance? |
| 6.5 Transition Novelty Comparison | State vs. transition novelty on Solaris | Does transition-level novelty improve over state-level novelty? |
| 6.6 Robotic Manipulation Tasks | FetchPickAndPlace, HandManipulateBlock (from Adventurer paper) | Does the metric bonus transfer to continuous control? |
| 6.7 Diagnostic Analysis | Latent distance, ensemble variance, bonus scale over training; comparison with Adventurer's pixel/feature novelty | How do the intrinsic reward dynamics differ between V1 and V4? |

**Key Papers**: Liu & Liu (2025) [Adventurer — experimental setup]; Wang et al. (2024) [EME — Atari/Minigrid/Robosuite/Habitat benchmarks]; Burda et al. (2019) [RND — hard exploration Atari]

**Missing Information**:
- **All experimental results must be generated.** The repository contains the complete implementation but no pre-computed experiment results (no `runs/` directory with metrics). The thesis requires:
  - V1–V4 comparison on at least 3 sparse-reward Atari games × 3 seeds
  - Ablation studies across hyperparameters
  - Diagnostic plots comparing novelty dynamics
- The Adventurer paper reports results on Montezuma's Revenge, Gravitar, Solaris, FetchPickAndPlace, and HandManipulateBlock; these should be replicated for V1 and then extended to V2–V4.
- EME's benchmarks include Minigrid, Robosuite, and Habitat; results on these would strengthen comparison with EME but are **not required** if Atari + MuJoCo results are sufficient.

---

### Chapter 7: Discussion

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 7.1 Summary of Findings | Recap: latent metric is efficient; EME scaling is informative only when normalised; V4 > V3 ≈ V2 under sparse rewards | — |
| 7.2 Relationship to Other Methods | How V4 relates to NGU, Agent57, RND, ICM; advantages and disadvantages | Where does the BiGAN latent metric sit among exploration strategies? |
| 7.3 Computational Cost Analysis | One encoder forward per step vs. full reconstruction (encoder + generator + discriminator); ensemble overhead | What is the computational tradeoff? |
| 7.4 Limitations | BiGAN training instability; ensemble variance collapse; frozen encoder prevents metric adaptation; evaluation only on limited benchmarks | What are the failure modes? |
| 7.5 Theoretical Considerations | BiGAN encoder inversion property; latent metric validity; scale-free property of normalised scaling; intrinsic reward vanishing | What theoretical guarantees can be established? |

**Key Papers**: All previously cited.

**Missing Information**:
- Formal convergence analysis of the BiGAN latent metric under the frozen-encoder regime.
- Comparison of computational cost (FLOPs, wall-clock time) between V1 and V4.

---

### Chapter 8: Conclusion and Future Work

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 8.1 Conclusion | Summarise contributions and key results | — |
| 8.2 Future Work | Parallel environments; learned $\alpha$ scheduling; alternative encoders (VAE, contrastive); hybrid V1+V4 bonuses; extending to model-based RL | What are the most promising directions? |

**Missing Information**: None — future work is speculative.

---

## 3. Key Papers Supporting Each Chapter

| Paper | Citation | Chapters |
|-------|----------|----------|
| Adventurer: Exploration with BiGAN for Deep RL | Liu & Liu (2025), Applied Intelligence | 1, 2, 3, 4, 5, 6, 7 |
| Effective Metric-Based Exploration Bonus (EME) | Wang et al. (2024), NeurIPS 2024 Spotlight | 1, 2, 4, 5, 6, 7 |
| Adversarial Feature Learning (BiGAN) | Donahue et al. (2017), ICLR 2017 | 2, 3, 4, 7 |
| ALI: Adversarially Learned Inference | Dumoulin et al. (2017), ICLR 2017 | 2, 3 |
| Proximal Policy Optimization Algorithms | Schulman et al. (2017), arXiv | 2, 3 |
| Trust Region Policy Optimization | Schulman et al. (2015), NIPS 2015 | 2 |
| Exploration by Random Network Distillation | Burda et al. (2019), ICLR 2019 | 2, 3, 6, 7 |
| Curiosity-Driven Exploration by Self-Supervised Prediction (ICM) | Pathak et al. (2017), ICML 2017 | 2, 7 |
| Generative Adversarial Exploration (GAEX) | Hong et al. (2019), DAI 2019 | 2, 7 |
| Never Give Up (NGU) | Badia et al. (2020a), ICLR 2020 | 2, 3, 7 |
| Agent57 | Badia et al. (2020b), arXiv 2020 | 2, 7 |
| Unifying Count-Based Exploration and Intrinsic Motivation | Bellemare et al. (2016), NIPS 2016 | 2 |
| Count-Based Exploration with NN Density Models | Ostrovski et al. (2017), ICML 2017 | 2 |
| Large Scale Adversarial Representation Learning (BigBiGAN) | Donahue & Simonyan (2019), ICLR 2019 | 2, 7 |
| Goodfellow et al. (2014) — GANs | Goodfellow et al. (2014), NIPS 2014 | 2 |
| Ecoffet et al. (2021) — Go-Explore | Ecoffet et al. (2021), ICML 2021 | 2, 3 |

---

## 4. Missing Information / Evidence Required Before Writing

| Item | Chapter(s) | Status | Action Required |
|------|-----------|--------|-----------------|
| Full EME paper (Wang et al., 2024) — theoretical derivation of bisimulation metric approximation and diversity-enhanced scaling factor | 2, 4, 5 | ⚠️ Not in repo | Download and cite the NeurIPS 2024 paper from OpenReview |
| Formal proof that latent discrepancy $\|E(s_t) - E(s_{t+1})\|_p$ is a valid metric on state space | 4, 7 | ❌ Not available | Derive from BiGAN encoder invertibility (Donahue et al., 2017, Theorem 3) or cite EME's theoretical framework |
| Formal proof that $\zeta / \mathbb{E}[\zeta]$ is scale-free | 5 | ❌ Not available | Straightforward derivation: if $r \to cr$ then $\hat{r}_k \to c\hat{r}_k$ (for linear models) so $\zeta \to c^2\zeta$ and $\mathbb{E}[\zeta] \to c^2\mathbb{E}[\zeta]$, hence the ratio is invariant |
| Convergence analysis of EMA reference level | 5 | ❌ Not available | Standard EMA convergence analysis can be referenced from time-series literature |
| Experimental results: V1–V4 comparison on hard Atari (Montezuma's Revenge, Gravitar, Solaris) × 3+ seeds | 6 | ❌ Not in repo | **Run experiments** using `run_metric_eme_comparison.sh` with appropriate hardware (GPU required for Atari) |
| Ablation study results (α, β, K, M, L1/L2, frozen encoder) | 6 | ❌ Not in repo | **Run experiments** with modified configurations |
| Computational cost comparison (FLOPs, wall-clock time) V1 vs. V4 | 7 | ❌ Not available | Profile both variants on the same hardware |
| Adventurer CIFAR-10 validation results | 6 | ⚠️ In Adventurer paper only | Reference from the published paper |
| Adventurer MuJoCo robotics results | 6 | ⚠️ In Adventurer paper only | Replicate for V1 or reference from published paper |
| Full Adventurer paper (for Appendix D: architecture details) | 3 | ⚠️ Available on arXiv | Download arXiv:2503.18612 |

---

## 5. Recommended Logical Order and Connections Between Chapters

```
Chapter 1 (Introduction)
    │
    │  defines problem, states contributions
    ▼
Chapter 2 (Background & Related Work)
    │
    │  provides all formal definitions and literature context
    │  ← feeds into every subsequent chapter
    ▼
Chapter 3 (Adventurer Framework)
    │
    │  establishes the baseline system that Chapter 4 modifies
    │  key equations: BiGAN objective, B(s), Eq. (5), two-stream PPO
    ▼
Chapter 4 (Metric-Based Exploration Bonus)
    │
    │  replaces B(s) with d_t · scaling
    │  introduces latent discrepancy (replacing reconstruction)
    │  introduces EME ensemble scaling (from Wang et al., 2024)
    │  ← depends on Chapter 3 for BiGAN encoder and Eq. (5)
    ▼
Chapter 5 (Normalised EME Scaling)
    │
    │  identifies V3 collapse problem (Chapter 4's scaling fails under sparse rewards)
    │  proposes V4 solution (normalised mode)
    │  ← extends Chapter 4's bonus formulation
    │  ← uses Chapter 3's Eq. (5) for bonus normalization
    ▼
Chapter 6 (Experimental Evaluation)
    │
    │  empirically validates all four variants
    │  ← Chapter 3 provides V1; Chapter 4 provides V2, V3; Chapter 5 provides V4
    │  ← ablations test components from Chapters 3–5
    ▼
Chapter 7 (Discussion)
    │
    │  interprets Chapter 6's results
    │  relates to Chapter 2's literature
    │  analyses limitations of Chapters 3–5
    ▼
Chapter 8 (Conclusion & Future Work)
```

### Critical Dependencies

1. **Chapter 3 → Chapter 4**: The latent discrepancy replaces reconstruction novelty; the BiGAN encoder trained by Adventurer's adversarial objective *defines* the latent metric. Without Chapter 3, Chapter 4's encoder has no training signal.

2. **Chapter 4 → Chapter 5**: The normalised EME scaling (Chapter 5) is a modification of the clamped EME scaling introduced in Chapter 4. The CartPole collapse demonstration (Section 5.5) directly compares V3 (Chapter 4) and V4 (Chapter 5).

3. **Chapters 3–5 → Chapter 6**: All four experiment variants (V1–V4) are defined across Chapters 3–5. Chapter 6 cannot be written until experimental results are generated.

4. **Chapter 2 → All**: Background definitions (MDPs, PPO, BiGAN, intrinsic rewards) are used throughout.

### Suggested Writing Order

1. **Chapter 2** first — establishes notation, definitions, and literature context that all other chapters reference.
2. **Chapter 3** second — defines the baseline system.
3. **Chapter 4** third — the core methodological contribution.
4. **Chapter 5** fourth — the key improvement and the thesis's central novelty.
5. **Chapter 1** fifth — now that all contributions are clear, write the introduction.
6. **Chapter 6** sixth — **requires experimental results**; write after running experiments.
7. **Chapter 7** seventh — interprets Chapter 6.
8. **Chapter 8** last — summarises everything.

---

## 6. Notation Convention (for consistency across chapters)

| Symbol | Meaning | First Appears |
|--------|---------|--------------|
| $s_t, a_t, r_t$ | State, action, reward at time $t$ | Ch. 2 |
| $E_\psi$ | BiGAN encoder with parameters $\psi$ | Ch. 3 |
| $G_\theta$ | BiGAN generator with parameters $\theta$ | Ch. 3 |
| $D_\omega$ | BiGAN joint discriminator with parameters $\omega$ | Ch. 3 |
| $L_G(s)$ | Pixel reconstruction error $\|s - G(E(s))\|_1$ | Ch. 3 |
| $L_D(s)$ | Feature matching error $\|f_D(s,E(s)) - f_D(G(E(s)),E(s))\|_1$ | Ch. 3 |
| $B(s)$ | Combined novelty score $\alpha L_G(s) + (1-\alpha) L_D(s)$ | Ch. 3 |
| $\tilde{n}(s)$ | Eq. (5) normalized novelty | Ch. 3 |
| $r_t^{ext}, r_t^{int}$ | Extrinsic and intrinsic reward streams | Ch. 3 |
| $\gamma, \lambda$ | Discount factor and GAE parameter | Ch. 3 |
| $\beta$ | Intrinsic advantage coefficient | Ch. 3 |
| $d_t$ | Latent state discrepancy $\|E(s_t) - E(s_{t+1})\|_p$ | Ch. 4 |
| $\zeta(r)$ | Ensemble reward variance | Ch. 4 |
| $M$ | Maximum reward scaling (upper clamp) | Ch. 4 |
| $K$ | Ensemble size | Ch. 4 |
| $b_t$ | Full exploration bonus $d_t \cdot \mathrm{scale}$ | Ch. 4 |
| $\mathbb{E}[\zeta]$ | EMA reference level for normalised mode | Ch. 5 |
| $m$ | EMA momentum (default 0.99) | Ch. 5 |
| $\zeta_\epsilon$ | Degeneracy threshold | Ch. 5 |

---

*This roadmap is based on verified information from the codebase (README.md, all source files, test files, and configuration) and the identified literature. No citations, results, equations, methods, or claims have been fabricated. Items that cannot be verified from the codebase or the searched literature are explicitly marked as "Missing Information" in Section 4.*
