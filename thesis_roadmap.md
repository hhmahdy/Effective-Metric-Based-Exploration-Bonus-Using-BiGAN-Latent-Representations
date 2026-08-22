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

### Chapter 2: Background

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| **2.1 Reinforcement Learning** | | |
| 2.1.1 Reinforcement Learning | Introduction to RL: agent–environment interaction, reward, episodes, the exploitation–exploration dilemma | Why and when do we use RL? What makes RL different from supervised/unsupervised learning? |
| 2.1.2 Markov Decision Processes and Policy Optimization | Formal MDP framework: states, actions, transitions, rewards, discount; value functions $V^\pi$, $Q^\pi$; policy gradient theorem | — |
| 2.1.3 Mirror Descent Policy Optimization (MDPO) | MD: first-order method in constrained convex optimization; trust-region policy update $\max_\pi \mathbb{E}_{s \sim \rho_{\pi_k}}[\mathbb{E}_{a \sim \pi_k}[A^{\pi_k}(s,a) \frac{\pi(a|s)}{\pi_k(a|s)}] - \frac{1}{t_k} D_\mathrm{KL}(\pi \| \pi_k)]$; on-policy and off-policy variants; multiple gradient steps vs. exact solve | How does MDPO balance stable policy updates with exploration? What is the relationship between MDPO, PPO, and TRPO? |
| **2.2 Exploration Methods** | | |
| 2.2.1 Exploration Methods and Comparison | Taxonomy of exploration strategies: uncertainty-oriented (UCB, posterior sampling, ensemble disagreement), intrinsic motivation-oriented (count-based, prediction-based, memory-based), and other notable approaches; systematic comparison across benchmarks; survey integration from Yang et al. (2021) and Ladosz et al. (2022) | What is the best way to compare between exploration methods and why? Where does each method excel or fail? |
| 2.3 Generative Adversarial Networks and BiGANs | Standard GANs, BiGAN joint adversarial game, encoder–generator inversion property (Theorem 3, Donahue et al. 2017) | Why does BiGAN learn an invertible mapping, and how does the encoder define a latent metric? |
| 2.4 Intrinsic Reward and Exploration Bonuses | Taxonomy: count-based, prediction-based, uncertainty-based, memory-based; formal definitions of each category | What are the failure modes of existing exploration methods in high-dimensional sparse-reward settings? |
| 2.5 Novelty-Based Exploration Methods | RND, ICM, GAEX, VAE-based methods; their relationship to reconstruction error | How does reconstruction novelty relate to visitation counts? |
| 2.6 Metric-Based Exploration and EME | Bisimulation metric, latent-space state discrepancy, EME's ensemble scaling factor | What are the limitations of current metric-based exploration bonuses, and how does EME address them? |
| 2.7 Episodic Memory and the Resettable Premise | Top-K episodic memory, NGU-style episodic+lifelong novelty, Agent57 | How does the resettable premise prevent intrinsic reward vanishing? |

**Key Papers**:
- Tomar et al. (2022) [MDPO — ICLR 2022]
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
- Yang et al. (2021) [Exploration Survey]
- Ladosz et al. (2022) [Exploration Survey]
- Bellemare et al. (2016) [Unifying Count-Based]
- Ostrovski et al. (2017) [Count-Based with PixelCNN]
- Goodfellow et al. (2014) [GANs]

**Missing Information**:
- The full MDPO paper (Tomar et al., 2022, ICLR) should be consulted for the precise mathematical derivation of on-policy and off-policy MDPO variants and their theoretical connections to TRPO and PPO.
- Yang et al. (2021) and Ladosz et al. (2022) surveys should be used as primary references for the systematic comparison in Section 2.2.1.

---

### Chapter 3: Related Work

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 3.1 Prediction-Based Exploration | Forward dynamics prediction (ICM, Stadie et al. 2015); random network distillation (RND, Burda et al. 2019); flow-based curiosity (FICM, Deng et al. 2020); strengths and failure modes (noisy TV problem) | How do prediction errors serve as novelty signals, and when do they fail? |
| 3.2 Generative-Model-Based Exploration | GAEX (Hong et al. 2019): discriminator scores as intrinsic rewards; VAE-based novelty (Asperti et al. 2021): reconstruction error; Adventurer (Liu & Liu 2025): BiGAN combined pixel+feature novelty | How do generative models estimate state novelty, and what is Adventurer's specific contribution? |
| 3.3 Count-Based and Pseudo-Count Methods | Count-based bonuses (Bellemare et al. 2016); PixelCNN density models (Ostrovski et al. 2017); hashing-based counts; relationship to Bayesian information gain | How do pseudo-counts connect to epistemic uncertainty? |
| 3.4 Uncertainty-Based Exploration | Ensemble disagreement (Pathak et al. 2019; Sekar et al. 2020); posterior sampling (Osband et al. 2016); Bayesian exploration bonuses; epistemic vs. aleatoric uncertainty decomposition | How does epistemic uncertainty drive exploration? |
| 3.5 Memory-Based Exploration | Episodic memory (NGU, Badia et al. 2020a); Go-Explore (Ecoffet et al. 2021); Agent57 (Badia et al. 2020b); resettable premise; lifelong + episodic novelty combination | How does memory-based exploration overcome intrinsic reward vanishing? |
| 3.6 Metric-Based Exploration | Bisimulation metric (Ferns et al. 2004; Zhang et al. 2020); latent-space state discrepancy; EME (Wang et al. 2024): robust metric + diversity-enhanced scaling factor | What theoretical and practical gaps exist in current metric-based exploration, and how does EME address them? |
| 3.7 Policy Optimization Landscape | TRPO (Schulman et al. 2015): constrained optimization with natural gradient + line search; PPO (Schulman et al. 2017): clipped surrogate objective; MDPO (Tomar et al. 2022): MD-based trust region with multiple SGD steps; comparison of stability, sample efficiency, and computational cost | How do TRPO, PPO, and MDPO differ in their trust-region enforcement, and what are the empirical tradeoffs? |
| 3.8 Positioning of This Work | How the proposed method relates to and extends prior work: Adventurer (base) → latent discrepancy (replaces reconstruction) → EME scaling (adds reward sensitivity) → normalised EME (fixes sparse-reward collapse) | Where does this work sit in the exploration landscape, and what gap does it fill? |

**Key Papers**:
- Pathak et al. (2017) [ICM]; Burda et al. (2019) [RND]; Hong et al. (2019) [GAEX]
- Liu & Liu (2025) [Adventurer]; Wang et al. (2024) [EME]
- Bellemare et al. (2016) [Pseudo-counts]; Ostrovski et al. (2017) [PixelCNN counts]
- Badia et al. (2020a) [NGU]; Badia et al. (2020b) [Agent57]; Ecoffet et al. (2021) [Go-Explore]
- Schulman et al. (2015) [TRPO]; Schulman et al. (2017) [PPO]; Tomar et al. (2022) [MDPO]
- Sekar et al. (2020) [Plan2Explore]; Osband et al. (2016) [Bootstrapped DQN]
- Ferns et al. (2004) [Bisimulation]; Zhang et al. (2020) [Bisimulation in DRL]

**Missing Information**:
- Ferns et al. (2004) and Zhang et al. (2020) should be consulted for the formal bisimulation metric definition used in EME's theoretical analysis.
- The full EME paper (Wang et al., 2024) should be consulted for its theoretical derivation of the bisimulation metric approximation and the formal derivation of the diversity-enhanced scaling factor.

---

### Chapter 4: The Adventurer Framework

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 4.1 BiGAN-Based State Novelty Estimation | Encoder $E_\psi$, generator $G_\theta$, discriminator $D_\omega$; joint adversarial objective | How does the BiGAN adversarial game induce a novelty estimator? |
| 4.2 Combined Novelty Score | $B(s) = \alpha L_G(s) + (1-\alpha)L_D(s)$; pixel reconstruction + feature matching | Why combine pixel-level and feature-level errors? What is the effect of $\alpha$? |
| 4.3 Intrinsic Reward Normalization | Eq. (5): reward-scale normalization $\tilde{n}(s) = (B(s) - \mu_B + \mu_{r^e}) / \sigma_B$ | How does normalization align intrinsic and extrinsic reward scales? |
| 4.4 Two-Stream PPO Integration | Separate extrinsic/intrinsic GAE, combined advantage $A_t = A_t^e + \beta A_t^i$ | Why separate reward streams rather than mixing rewards before GAE? |
| 4.5 Episodic Memory and Resettable Exploration | Algorithm 2: top-K novel state memory, state restoration | How does the resettable premise double exploration performance? |
| 4.6 Transition Novelty Variant | Latent forward model $f_\phi$, transition novelty $T(s_t, a_t, s_{t+1})$ | Can transition-level novelty improve over state-level novelty? |
| 4.7 Implementation Architecture | Network architectures, training schedule, replay buffers | — |

**Key Papers**: Liu & Liu (2025) [Adventurer]; Donahue et al. (2017) [BiGAN]; Burda et al. (2019) [RND — two value heads]; Schulman et al. (2017) [PPO]

**Missing Information**: The Adventurer paper's CIFAR-10 validation experiment (Section 5.1.1) provides empirical evidence that BiGAN novelty scores correlate with state novelty; this should be referenced if the full paper is available.

---

### Chapter 5: Metric-Based Exploration Bonus with BiGAN Latent Representations

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 5.1 Motivation and Problem Analysis | Reconstruction novelty is expensive (requires G and D passes); not transition-aware; does not distinguish reward-relevant from reward-irrelevant novelty | What are the computational and conceptual limitations of Adventurer's reconstruction novelty? |
| 5.2 Latent State Discrepancy | $d_t = \|E_\psi(s_t) - E_\psi(s_{t+1})\|_p$; $p \in \{1, 2\}$; one encoder forward per step vs. full reconstruction pipeline | How does the latent metric capture transition novelty without reconstruction? |
| 5.3 EME Diversity-Enhanced Scaling | Ensemble of $K$ bootstrapped reward models; $\zeta(r) = \mathrm{Var}_k(\hat{r}_k(s_{t+1}))$; scaling factor $\min(\max(\zeta, 1), M)$ | How does the ensemble variance identify reward-relevant but poorly modelled regions? |
| 5.4 The Full Exploration Bonus | $b_t = d_t \cdot \min(\max(\zeta(r), 1), M)$; integration with Adventurer's Eq. (5) normalization | How do the metric term and scaling factor interact? |
| 5.5 Implementation Details | `LatentStateDiscrepancy`, `EnsembleRewardVariance`, `MetricIntrinsicReward`; bootstrap diversity; frozen encoder option | — |

**Key Papers**: Wang et al. (2024) [EME]; Liu & Liu (2025) [Adventurer]; Donahue et al. (2017) [BiGAN — encoder as feature representation]

**Missing Information**: A formal theoretical analysis proving that the latent discrepancy $d_t$ is a valid metric on state space (given sufficient encoder capacity) would strengthen this chapter. This is **not** present in the codebase and would need to be derived or cited from EME's theoretical framework.

---

### Chapter 6: Normalised EME Scaling for Sparse Rewards

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 6.1 The Sparse-Reward Collapse Problem | In sparse Atari rewards, almost all regression targets are zero, so ensemble members agree, $\zeta \ll 1$, the lower clamp binds, and V3 collapses to V2 | Why does the published EME scaling factor fail in sparse-reward settings? |
| 6.2 Normalised Scaling Mode | $\mathrm{scale} = \min(\zeta / \mathbb{E}[\zeta], M)$; EMA reference level $\mathbb{E}[\zeta]$ with momentum; scale-free property | How does normalisation restore the scaling factor's discriminative power at any reward magnitude? |
| 6.3 Degeneracy Handling | When $\mathbb{E}[\zeta] < \zeta_\epsilon$, the ensemble carries no usable signal; fall back to scale = 1 (pure metric bonus) | How does the normalised mode degrade gracefully rather than zeroing exploration? |
| 6.4 Reference Level as Exponential Moving Average | $\mathbb{E}[\zeta] \leftarrow m \cdot \mathbb{E}[\zeta] + (1-m) \cdot \mathrm{mean}(\zeta_\mathrm{batch})$; seeded by first batch; why EMA over cumulative mean | Why use an EMA rather than a cumulative mean for the reference level? |
| 6.5 Empirical Demonstration of the Collapse | CartPole 512-step data: raw $\zeta$ rises from 4.6e-5 to 7.5e-3; V3 scale is constant at 1.0; V4 scale varies (2.68 → 3.41 → 3.40 → 2.74) | What does the empirical data show about V3 vs. V4? |
| 6.6 Frozen Encoder for Stationary Metrics | `--freeze-encoder-after-updates N`; BiGAN encoder stops receiving gradients; generator and discriminator continue | Why freeze the encoder to stabilise the latent metric? |

**Key Papers**: Wang et al. (2024) [EME — clamped scaling]; Liu & Liu (2025) [Adventurer — Eq. (5) normalization]

**Missing Information**:
- A **formal proof** that $\zeta / \mathbb{E}[\zeta]$ is scale-free (i.e., invariant under rescaling of the reward) would strengthen Section 6.2. The codebase demonstrates this empirically but does not prove it.
- **Large-scale experimental results** comparing V3 and V4 on hard Atari games (Montezuma's Revenge, Gravitar, Solaris) are described in the README but **not included in the repository**. These results are critical for the thesis and must be generated.
- A **theoretical analysis** of the EMA reference level's convergence properties would strengthen Section 6.4.

---

### Chapter 7: Experimental Evaluation

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 7.1 Experimental Setup | Environments, hyperparameters, training budget, evaluation protocol | — |
| 7.2 BiGAN Validation | CIFAR-10 novelty estimation (from Adventurer paper); Montezuma's Revenge novelty ranking | Does BiGAN accurately estimate state novelty? |
| 7.3 V1–V4 Variant Comparison on Sparse-Reward Atari | Montezuma's Revenge, Gravitar, Solaris; game score, intrinsic reward, total reward | Does V4 outperform V3 on sparse-reward games? Is V3 ≈ V2 where $\zeta < 1$? |
| 7.4 Ablation Studies | $\alpha$ sweep; $\beta$ sweep; ensemble size $K$; max reward scaling $M$; latent norm ($L_1$ vs. $L_2$); frozen vs. continually trained encoder; normalised vs. clamped mode | Which factors contribute most to V4's performance? |
| 7.5 Transition Novelty Comparison | State vs. transition novelty on Solaris | Does transition-level novelty improve over state-level novelty? |
| 7.6 Robotic Manipulation Tasks | FetchPickAndPlace, HandManipulateBlock (from Adventurer paper) | Does the metric bonus transfer to continuous control? |
| 7.7 Diagnostic Analysis | Latent distance, ensemble variance, bonus scale over training; comparison with Adventurer's pixel/feature novelty | How do the intrinsic reward dynamics differ between V1 and V4? |

**Key Papers**: Liu & Liu (2025) [Adventurer — experimental setup]; Wang et al. (2024) [EME — Atari/Minigrid/Robosuite/Habitat benchmarks]; Burda et al. (2019) [RND — hard exploration Atari]

**Missing Information**:
- **All experimental results must be generated.** The repository contains the complete implementation but no pre-computed experiment results (no `runs/` directory with metrics). The thesis requires:
  - V1–V4 comparison on at least 3 sparse-reward Atari games × 3 seeds
  - Ablation studies across hyperparameters
  - Diagnostic plots comparing novelty dynamics
- The Adventurer paper reports results on Montezuma's Revenge, Gravitar, Solaris, FetchPickAndPlace, and HandManipulateBlock; these should be replicated for V1 and then extended to V2–V4.
- EME's benchmarks include Minigrid, Robosuite, and Habitat; results on these would strengthen comparison with EME but are **not required** if Atari + MuJoCo results are sufficient.

---

### Chapter 8: Discussion

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 8.1 Summary of Findings | Recap: latent metric is efficient; EME scaling is informative only when normalised; V4 > V3 ≈ V2 under sparse rewards | — |
| 8.2 Relationship to Other Methods | How V4 relates to NGU, Agent57, RND, ICM, MDPO-based exploration; advantages and disadvantages | Where does the BiGAN latent metric sit among exploration strategies? |
| 8.3 Computational Cost Analysis | One encoder forward per step vs. full reconstruction (encoder + generator + discriminator); ensemble overhead | What is the computational tradeoff? |
| 8.4 Limitations | BiGAN training instability; ensemble variance collapse; frozen encoder prevents metric adaptation; evaluation only on limited benchmarks | What are the failure modes? |
| 8.5 Theoretical Considerations | BiGAN encoder inversion property; latent metric validity; scale-free property of normalised scaling; intrinsic reward vanishing | What theoretical guarantees can be established? |

**Key Papers**: All previously cited.

**Missing Information**:
- Formal convergence analysis of the BiGAN latent metric under the frozen-encoder regime.
- Comparison of computational cost (FLOPs, wall-clock time) between V1 and V4.

---

### Chapter 9: Conclusion and Future Work

| Section | Purpose | Research Questions |
|---------|---------|-------------------|
| 9.1 Conclusion | Summarise contributions and key results | — |
| 9.2 Future Work | Parallel environments; learned $\alpha$ scheduling; alternative encoders (VAE, contrastive); hybrid V1+V4 bonuses; extending to model-based RL; MDPO as alternative policy optimizer to PPO | What are the most promising directions? |

**Missing Information**: None — future work is speculative.

---

## 3. Key Papers Supporting Each Chapter

| Paper | Citation | Chapters |
|-------|----------|----------|
| Adventurer: Exploration with BiGAN for Deep RL | Liu & Liu (2025), Applied Intelligence | 1, 2, 3, 4, 5, 6, 7, 8 |
| Effective Metric-Based Exploration Bonus (EME) | Wang et al. (2024), NeurIPS 2024 Spotlight | 1, 2, 3, 5, 6, 7, 8 |
| Mirror Descent Policy Optimization (MDPO) | Tomar et al. (2022), ICLR 2022 | 2, 3, 8, 9 |
| Adversarial Feature Learning (BiGAN) | Donahue et al. (2017), ICLR 2017 | 2, 3, 4, 5, 8 |
| ALI: Adversarially Learned Inference | Dumoulin et al. (2017), ICLR 2017 | 2, 3 |
| Proximal Policy Optimization Algorithms | Schulman et al. (2017), arXiv | 2, 3, 4 |
| Trust Region Policy Optimization | Schulman et al. (2015), NIPS 2015 | 2, 3 |
| Exploration by Random Network Distillation | Burda et al. (2019), ICLR 2019 | 1, 2, 3, 4, 6, 7, 8 |
| Curiosity-Driven Exploration by Self-Supervised Prediction (ICM) | Pathak et al. (2017), ICML 2017 | 2, 3, 8 |
| Generative Adversarial Exploration (GAEX) | Hong et al. (2019), DAI 2019 | 2, 3, 8 |
| Never Give Up (NGU) | Badia et al. (2020a), ICLR 2020 | 2, 3, 8 |
| Agent57 | Badia et al. (2020b), arXiv 2020 | 2, 3, 8 |
| Go-Explore | Ecoffet et al. (2021), ICML 2021 | 2, 3 |
| Unifying Count-Based Exploration and Intrinsic Motivation | Bellemare et al. (2016), NIPS 2016 | 2, 3 |
| Count-Based Exploration with NN Density Models | Ostrovski et al. (2017), ICML 2017 | 2, 3 |
| Large Scale Adversarial Representation Learning (BigBiGAN) | Donahue & Simonyan (2019), ICLR 2019 | 2, 3 |
| Goodfellow et al. (2014) — GANs | Goodfellow et al. (2014), NIPS 2014 | 2 |
| Exploration in DRL: A Comprehensive Survey | Yang et al. (2021), arXiv | 2, 3 |
| Exploration in DRL: A Survey | Ladosz et al. (2022), arXiv | 2, 3 |
| Policy Optimization with Stochastic Mirror Descent | Yang & Zhang (2019), arXiv | 2 |
| Plan2Explore | Sekar et al. (2020), ICML 2020 | 3 |
| Bootstrapped DQN | Osband et al. (2016), NIPS 2016 | 3 |
| Bisimulation Metrics | Ferns et al. (2004) | 3 |
| Bisimulation in DRL | Zhang et al. (2020) | 3 |

---

## 4. Missing Information / Evidence Required Before Writing

| Item | Chapter(s) | Status | Action Required |
|------|-----------|--------|-----------------|
| Full EME paper (Wang et al., 2024) — theoretical derivation of bisimulation metric approximation and diversity-enhanced scaling factor | 2, 3, 5, 6 | ⚠️ Not in repo | Download and cite the NeurIPS 2024 paper from OpenReview |
| Full MDPO paper (Tomar et al., 2022) — precise mathematical derivation and TRPO/PPO connections | 2, 3 | ⚠️ Not in repo | Download and cite the ICLR 2022 paper from OpenReview |
| Exploration survey papers (Yang et al. 2021; Ladosz et al. 2022) — systematic comparison tables and benchmark results | 2, 3 | ⚠️ Not in repo | Download from arXiv |
| Formal proof that latent discrepancy $\|E(s_t) - E(s_{t+1})\|_p$ is a valid metric on state space | 5, 8 | ❌ Not available | Derive from BiGAN encoder invertibility (Donahue et al., 2017, Theorem 3) or cite EME's theoretical framework |
| Formal proof that $\zeta / \mathbb{E}[\zeta]$ is scale-free | 6 | ❌ Not available | Straightforward derivation: if $r \to cr$ then $\hat{r}_k \to c\hat{r}_k$ (for linear models) so $\zeta \to c^2\zeta$ and $\mathbb{E}[\zeta] \to c^2\mathbb{E}[\zeta]$, hence the ratio is invariant |
| Convergence analysis of EMA reference level | 6 | ❌ Not available | Standard EMA convergence analysis can be referenced from time-series literature |
| Experimental results: V1–V4 comparison on hard Atari (Montezuma's Revenge, Gravitar, Solaris) × 3+ seeds | 7 | ❌ Not in repo | **Run experiments** using `run_metric_eme_comparison.sh` with appropriate hardware (GPU required for Atari) |
| Ablation study results (α, β, K, M, L1/L2, frozen encoder) | 7 | ❌ Not in repo | **Run experiments** with modified configurations |
| Computational cost comparison (FLOPs, wall-clock time) V1 vs. V4 | 8 | ❌ Not available | Profile both variants on the same hardware |
| Adventurer CIFAR-10 validation results | 7 | ⚠️ In Adventurer paper only | Reference from the published paper |
| Adventurer MuJoCo robotics results | 7 | ⚠️ In Adventurer paper only | Replicate for V1 or reference from published paper |
| Full Adventurer paper (for Appendix D: architecture details) | 4 | ⚠️ Available on arXiv | Download arXiv:2503.18612 |
| Bisimulation metric references (Ferns et al. 2004; Zhang et al. 2020) | 3 | ⚠️ Not in repo | Locate and cite |

---

## 5. Recommended Logical Order and Connections Between Chapters

```
Chapter 1 (Introduction)
    │
    │  defines problem, states contributions
    ▼
Chapter 2 (Background)
    │
    │  provides all formal definitions: RL, MDPs, MDPO,
    │  exploration taxonomy, GANs/BiGANs, intrinsic rewards,
    │  metric-based exploration, episodic memory
    │  ← feeds into every subsequent chapter
    ▼
Chapter 3 (Related Work)
    │
    │  surveys and positions all prior work in detail
    │  compares methods systematically
    │  ← depends on Ch. 2 definitions for consistent analysis
    ▼
Chapter 4 (Adventurer Framework)
    │
    │  establishes the baseline system that Chapter 5 modifies
    │  key equations: BiGAN objective, B(s), Eq. (5), two-stream PPO
    │  ← depends on Ch. 2 for BiGAN and PPO background
    │  ← references Ch. 3 for comparison with other generative-model methods
    ▼
Chapter 5 (Metric-Based Exploration Bonus)
    │
    │  replaces B(s) with d_t · scaling
    │  introduces latent discrepancy (replacing reconstruction)
    │  introduces EME ensemble scaling (from Wang et al., 2024)
    │  ← depends on Ch. 4 for BiGAN encoder and Eq. (5)
    │  ← references Ch. 3 for EME's theoretical framework
    ▼
Chapter 6 (Normalised EME Scaling)
    │
    │  identifies V3 collapse problem (Chapter 5's scaling fails under sparse rewards)
    │  proposes V4 solution (normalised mode)
    │  ← extends Chapter 5's bonus formulation
    │  ← uses Chapter 4's Eq. (5) for bonus normalization
    ▼
Chapter 7 (Experimental Evaluation)
    │
    │  empirically validates all four variants
    │  ← Chapter 4 provides V1; Chapter 5 provides V2, V3; Chapter 6 provides V4
    │  ← ablations test components from Chapters 4–6
    ▼
Chapter 8 (Discussion)
    │
    │  interprets Chapter 7's results
    │  relates to Chapter 3's literature
    │  analyses limitations of Chapters 4–6
    ▼
Chapter 9 (Conclusion & Future Work)
```

### Critical Dependencies

1. **Chapter 2 → Chapter 3**: Background provides the formal definitions and notation that Related Work uses for analysis and comparison.

2. **Chapter 3 → Chapter 4**: Related Work establishes the landscape of existing methods, making clear what Adventurer contributes and where it falls short.

3. **Chapter 4 → Chapter 5**: The latent discrepancy replaces reconstruction novelty; the BiGAN encoder trained by Adventurer's adversarial objective *defines* the latent metric. Without Chapter 4, Chapter 5's encoder has no training signal.

4. **Chapter 5 → Chapter 6**: The normalised EME scaling (Chapter 6) is a modification of the clamped EME scaling introduced in Chapter 5. The CartPole collapse demonstration (Section 6.5) directly compares V3 (Chapter 5) and V4 (Chapter 6).

5. **Chapters 4–6 → Chapter 7**: All four experiment variants (V1–V4) are defined across Chapters 4–6. Chapter 7 cannot be written until experimental results are generated.

6. **Chapter 2 → All**: Background definitions (MDPs, PPO/MDPO, BiGAN, intrinsic rewards) are used throughout.

### Suggested Writing Order

1. **Chapter 2** first — establishes notation, definitions, and formal background that all other chapters reference.
2. **Chapter 3** second — surveys prior work using the framework established in Chapter 2.
3. **Chapter 4** third — defines the baseline system.
4. **Chapter 5** fourth — the core methodological contribution.
5. **Chapter 6** fifth — the key improvement and the thesis's central novelty.
6. **Chapter 1** sixth — now that all contributions are clear, write the introduction.
7. **Chapter 7** seventh — **requires experimental results**; write after running experiments.
8. **Chapter 8** eighth — interprets Chapter 7.
9. **Chapter 9** last — summarises everything.

---

## 6. Notation Convention (for consistency across chapters)

| Symbol | Meaning | First Appears |
|--------|---------|--------------|
| $s_t, a_t, r_t$ | State, action, reward at time $t$ | Ch. 2 |
| $\pi, \pi_k$ | Policy, policy at iteration $k$ | Ch. 2 |
| $V^\pi, Q^\pi$ | State-value and action-value functions | Ch. 2 |
| $A^\pi$ | Advantage function | Ch. 2 |
| $D_\mathrm{KL}$ | KL divergence | Ch. 2 |
| $E_\psi$ | BiGAN encoder with parameters $\psi$ | Ch. 4 |
| $G_\theta$ | BiGAN generator with parameters $\theta$ | Ch. 4 |
| $D_\omega$ | BiGAN joint discriminator with parameters $\omega$ | Ch. 4 |
| $L_G(s)$ | Pixel reconstruction error $\|s - G(E(s))\|_1$ | Ch. 4 |
| $L_D(s)$ | Feature matching error $\|f_D(s,E(s)) - f_D(G(E(s)),E(s))\|_1$ | Ch. 4 |
| $B(s)$ | Combined novelty score $\alpha L_G(s) + (1-\alpha) L_D(s)$ | Ch. 4 |
| $\tilde{n}(s)$ | Eq. (5) normalized novelty | Ch. 4 |
| $r_t^{ext}, r_t^{int}$ | Extrinsic and intrinsic reward streams | Ch. 4 |
| $\gamma, \lambda$ | Discount factor and GAE parameter | Ch. 4 |
| $\beta$ | Intrinsic advantage coefficient | Ch. 4 |
| $d_t$ | Latent state discrepancy $\|E(s_t) - E(s_{t+1})\|_p$ | Ch. 5 |
| $\zeta(r)$ | Ensemble reward variance | Ch. 5 |
| $M$ | Maximum reward scaling (upper clamp) | Ch. 5 |
| $K$ | Ensemble size | Ch. 5 |
| $b_t$ | Full exploration bonus $d_t \cdot \mathrm{scale}$ | Ch. 5 |
| $\mathbb{E}[\zeta]$ | EMA reference level for normalised mode | Ch. 6 |
| $m$ | EMA momentum (default 0.99) | Ch. 6 |
| $\zeta_\epsilon$ | Degeneracy threshold | Ch. 6 |

---

*This roadmap is based on verified information from the codebase (README.md, all source files, test files, and configuration) and the identified literature. No citations, results, equations, methods, or claims have been fabricated. Items that cannot be verified from the codebase or the searched literature are explicitly marked as "Missing Information" in Section 4.*
