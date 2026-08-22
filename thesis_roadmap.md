# Thesis Roadmap

## Effective Metric-Based Exploration Bonus Using BiGAN Latent Representations

---

## 1. Executive Overview & Problem Formulation

### 1.1 Research Scope and Context
This thesis investigates **sample-efficient exploration in deep reinforcement learning (RL)** for environments characterized by high-dimensional visual observations and sparse extrinsic rewards. In such environments, standard $\epsilon$-greedy and entropy-regularized exploration mechanisms fail because the probability of randomly encountering a reward signal decays exponentially with horizon length and state dimensionality.

The research extends the **Adventurer** framework [(Liu & Liu, 2025)](https://arxiv.org/abs/2503.18612)—which leverages Bidirectional Generative Adversarial Networks (BiGANs) to estimate state novelty—by substituting its computationally intensive, pixel/feature reconstruction-error exploration bonus with a **metric-based exploration bonus** computed directly in the latent representation space of the BiGAN encoder. This latent transition metric is scaled by the epistemic disagreement (variance) of an ensemble of bootstrapped reward predictors, following the **Effective Metric-based Exploration-bonus (EME)** framework [(Wang et al., 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/6a39cf3b666f8bdb2223f253981f3869-Paper-Conference.pdf).

```
                  HIGH-LEVEL CONCEPTUAL ARCHITECTURE
                  
  State Observation s_t ───┐
                           ├─► BiGAN Encoder E_ψ ──► Latent State z_t ──┐
  Next State s_{t+1}   ───┘                                             │
                                                                        ▼
                                                          Latent Discrepancy Metric
                                                          d_t = ||z_t - z_{t+1}||_p
                                                                        │
                                                                        ▼
  Ensemble Reward Models                                      Multiplicative Bonus
  {r̂_1, ..., r̂_K}(s_{t+1}) ──► Ensemble Variance ζ(r) ──────► b_t = d_t · Scale(ζ)
                                                                        │
                                                                        ▼
                                                           Two-Stream Policy Optimizer
                                                               (PPO / MDPO)
```

---

### 1.2 Identified Theoretical and Practical Gaps
1. **Adventurer Reconstruction Inefficiency:** Adventurer's novelty metric $B(s) = \alpha L_G(s) + (1-\alpha)L_D(s)$ requires executing full Generator ($G_\theta$) and Discriminator ($D_\omega$) forward passes at every environment step. Furthermore, reconstruction error measures static state unfamiliarity rather than dynamic transition progress, and it is prone to the "noisy TV" problem where task-irrelevant visual entropy yields persistent novelty.
2. **EME Clamping Collapse under Sparse Rewards:** Published EME scales the metric bonus via a clamped variance factor $\text{scale} = \min(\max(\zeta(r), 1), M)$. In sparse-reward domains (e.g., hard exploration Atari games like *Montezuma's Revenge*, *Solaris*, *Gravitar*), nearly all observed extrinsic rewards are zero. The ensemble predictions rapidly converge to zero ($\hat{r}_k \approx 0$), causing the raw variance $\zeta(r) \ll 1$. Consequently, the lower clamp at $1.0$ permanently binds, stripping the bonus of all reward-sensitive modulation and causing EME to degenerate to a naive unweighted metric bonus.
3. **Representation Metric Stationarity:** Continuous training of the BiGAN encoder induces non-stationarity in the induced latent metric space, which can destabilize value function learning in the downstream policy optimizer.

---

### 1.3 Core Contributions (Verified in Codebase)

1. **Latent State Discrepancy Formulation (`exploration/state_discrepancy.py`):**
   $$d_t = \|E_\psi(s_t) - E_\psi(s_{t+1})\|_p, \quad p \in \{1, 2\}$$
   Eliminates full generator/discriminator reconstruction overhead during policy rollouts while inheriting the structural invertibility properties of the BiGAN joint adversarial game [(Donahue et al., 2017)](https://openreview.net/forum?id=Byk-VI9eg).

2. **Diversity-Enhanced Ensemble Scaling (`exploration/ensemble_scaling.py`):**
   $$\zeta(r_{s_t}^{a_t}) = \frac{1}{K}\sum_{k=1}^K \left(\hat{r}_k(s_t, a_t) - \bar{r}(s_t, a_t)\right)^2$$
   Dynamically amplifies exploration bonuses in state transitions that exhibit reward relevance under epistemic model disagreement.

3. **Scale-Free Normalized EME Scaling Mode (`exploration/intrinsic_reward.py`):**
   $$\text{scale}_t = \min\left(\frac{\zeta_t}{\mathbb{E}[\zeta]_t + \epsilon}, M\right)$$
   Maintains discriminative scaling across arbitrary reward magnitudes and includes a graceful degeneracy fallback to pure metric exploration ($\text{scale} = 1.0$) when ensemble variance drops below a numerical floor $\zeta_\epsilon$.

4. **Systematic Four-Variant Comparative Taxonomy:**
   * **V1 (Baseline):** Original Adventurer reconstruction novelty $B(s)$.
   * **V2:** Latent state discrepancy only ($\text{scale} \equiv 1.0$).
   * **V3:** Latent state discrepancy + clamped EME scaling ($\min(\max(\zeta, 1), M)$).
   * **V4 (Proposed):** Latent state discrepancy + scale-free EMA-normalized EME scaling ($\min(\zeta / \mathbb{E}[\zeta], M)$).

---

## 2. Structural Standards and Chapter Boundaries

To prevent structural imbalances, this thesis enforces strict chapter demarcations:

```
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │                           CHAPTER ROLE TAXONOMY                             │
  ├───────────────────┬─────────────────────────────────────────────────────────┤
  │ Chapter 2         │ DEFINES & FORMALIZES: Core mathematical principles,     │
  │ (Background)      │ foundational theorems, and category taxonomies.         │
  ├───────────────────┼─────────────────────────────────────────────────────────┤
  │ Chapter 3         │ COMPARES & POSITIONS: Methodological landscape, trade-  │
  │ (Related Work)    │ offs, failure modes, and benchmark contextualization.   │
  ├───────────────────┼─────────────────────────────────────────────────────────┤
  │ Chapter 4         │ BASELINE ARCHITECTURE: Formal specification of the      │
  │ (Adventurer)      │ un-modified Adventurer BiGAN exploration pipeline.      │
  ├───────────────────┼─────────────────────────────────────────────────────────┤
  │ Chapters 5 & 6    │ NOVEL METHODOLOGY: Latent metric formulation (Ch. 5)    │
  │ (Proposed Method) │ and normalized scaling derivation (Ch. 6).              │
  ├───────────────────┼─────────────────────────────────────────────────────────┤
  │ Chapter 7         │ EMPIRICAL EVALUATION: Rigorous benchmark experiments,   │
  │ (Experiments)     │ multi-seed statistics, ablations, and diagnostics.      │
  ├───────────────────┼─────────────────────────────────────────────────────────┤
  │ Chapter 8 & 9     │ SYNTHESIS & OUTLOOK: Theoretical limits, computational  │
  │ (Discussion/Concl)│ profiling, architectural boundaries, and future work.   │
  └───────────────────┴─────────────────────────────────────────────────────────┘
```

* **Global Equation Numbering:** Equations indexed sequentially across chapters.
* **Statistical Error Reporting:** All results must report mean ± 95% CI across at least $N \ge 3$ distinct random seeds.

---

## 3. Proposed Chapters, Sections, and Detailed Information

### Chapter 1: Introduction *(8–12 pages)*

* **1.1 Motivation & Context:** The fundamental exploration bottleneck in high-dimensional Deep RL; limitations of $\epsilon$-greedy and Gaussian action noise in sparse-reward visual domains; high-level concept of intrinsic motivation and curiosity-driven exploration.
* **1.2 Problem Statement:** Mathematical formulation of the sparse-reward exploration dilemma in MDPs; specific failures of reconstruction-based novelty (computational bottlenecks, "noisy TV"); failure of existing metric scaling factors (EME) under near-zero reward statistics.
* **1.3 Summary of Contributions:** Concise itemization of the four methodological contributions.
* **1.4 Thesis Organization:** Structural walkthrough of Chapters 2–9 and Appendices.

---

### Chapter 2: Background & Theoretical Foundations *(25–30 pages)*

> **Chapter 2 vs. Chapter 3 boundary**: Chapter 2 **defines and formalises** concepts. Chapter 3 **compares, evaluates, and positions** methods. No systematic benchmark comparisons in Chapter 2.

```
  2.1 Reinforcement Learning & MDPs
      ├── 2.1.1 Foundations & Policy Optimization
      ├── 2.1.2 Generalized Advantage Estimation (GAE)
      └── 2.1.3 Mirror Descent Policy Optimization (MDPO), PPO, and TRPO
  2.2 The Exploration Problem in Deep RL
  2.3 Intrinsic Motivation & Novelty Taxonomies
  2.4 Generative Adversarial Networks & BiGAN Invertibility
  2.5 Novelty-Based Exploration Formulations (RND, ICM, GAEX)
  2.6 Metric-Based Exploration & Theoretical EME Bounds
  2.7 Episodic Memory & The Resettable Exploration Premise
```

* **2.1.1 Formal MDP Framework:** $\mathcal{M} = \langle \mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma, \rho_0 \rangle$; $V^\pi(s)$, $Q^\pi(s,a)$, $A^\pi(s,a)$, Policy Gradient Theorem.
* **2.1.2 Generalized Advantage Estimation:** $\text{GAE}(\gamma, \lambda)$ derivation for variance reduction.
* **2.1.3 MDPO:** Bregman divergence-regularized trust-region optimization; MDPO–PPO–TRPO theoretical relationships; empirical tradeoffs.
* **2.2 The Exploration Problem:** Exploration-exploitation trade-off; sample complexity in visual state spaces; curse of dimensionality.
* **2.3 Intrinsic Motivation Framework:** $r_t^{\text{total}} = r_t^{\text{ext}} + \beta r_t^{\text{int}}$; formal categories: Count-based, Prediction error, Epistemic uncertainty, Metric-based.
* **2.4 GANs & BiGANs:** Standard GAN minimax; BiGAN/ALI joint adversarial objective; **BiGAN Invertibility Theorem (Theorem 3, Donahue et al., 2017):** $E = G^{-1}$ a.e. at global optimum.
* **2.5 Novelty-Based Formulations:** RND, ICM, GAEX formal definitions.
* **2.6 Metric-Based Exploration & EME Foundations:** Classical Bisimulation Metric (Ferns et al., 2004); **EME Metric (Definition 2, Eq. 4)**; **Theorem 1 (Fixed-Point Convergence)**; **Theorem 2 (Value Difference Bound)**; **Propositions 1–3**; Ensemble Variance (Eq. 8); Diversity-Enhanced Scaling (Eq. 10); Tractable Loss (Eq. 9); Method Comparison Table 1.
* **2.7 Episodic Memory:** Lifelong vs. episodic novelty; Top-$K$ buffers; preventing exploration vanishing.

**Verified Paper Details (extracted and recorded in repository)**

All verified paper details (EME, MDPO, LIBERTY, Ferns bisimulation, surveys, Kayal, CIM) are documented in the repository commit history and the previous version of this roadmap. Key verified items:

| Paper | Source | Status |
|-------|--------|--------|
| EME (Wang et al., 2024) | OpenReview QpKWFLtZKi / NeurIPS PDF | ✅ Theorems 1–2, Props 1–3, Def. 2, Eqs. 4/8/9/10, Table 1 |
| MDPO (Tomar et al., 2022) | arXiv:2005.09814 | ✅ MD update rules (Eqs. 4–5), convergence rates, TRPO/PPO/SAC connections |
| LIBERTY (Wang et al., 2023) | NeurIPS 2023 PDF / OpenReview 0FhKURbTyF | ✅ Potential-based bonus, Theorems 1–2, EME's critique (Props 1–2) |
| Ferns et al. (2004) | arXiv:1207.4114 | ✅ Bisimulation metrics, value function bound |
| Yang et al. (2021) | arXiv:2109.06668 | ✅ Uncertainty/intrinsic motivation taxonomy |
| Amin et al. (2021) | arXiv:2109.00157 | ✅ Undirected/directed taxonomy |
| Ladosz et al. (2022) | arXiv:2205.00824 | ✅ 7-category taxonomy |
| Kayal et al. (2025) | Neural Comput & Applic 37 | ✅ 4-level diversity taxonomy (State/State+Dynamics/Policy/Skill) |
| CIM (Zheng et al., 2024) | arXiv:2407.09247 / IJCAI 2024 | ✅ Constrained intrinsic motivation, Lagrangian adaptive scaling |

---

### Chapter 3: Related Work & Comparative Taxonomy *(20–25 pages)*

> **Chapter 3's role**: Surveys, compares, and positions methods within the taxonomy from Chapter 2.

```
  3.1 Prediction-Based Curiosity (ICM, RND, Flow-based)
  3.2 Generative-Model Novelty (GAEX, VAE, BiGAN)
  3.3 Count-Based & Density-Based Methods (PixelCNN, Hashing)
  3.4 Epistemic Uncertainty & Ensemble Disagreement (Plan2Explore, Bootstrapped DQN)
  3.5 Memory-Based Exploration (NGU, Agent57, Go-Explore)
  3.6 Bisimulation & Metric-Based Exploration (LIBERTY, RIDE, EME)
  3.7 Information-Theoretic Exploration (VIME, Empowerment)
  3.8 Systematic Positioning & Trade-Off Matrix
```

* **3.6** includes: LIBERTY as EME precursor; EME's Propositions 1–2 addressing LIBERTY's $W_2$ relaxation and shifted distance; comparison with RIDE on scalability/approximation gap.
* **3.8** includes: comparison of adaptive intrinsic reward scaling (CIM Lagrangian vs. this thesis's EMA normalisation); Kayal et al. (2025) diversity-level taxonomy positioning latent discrepancy at State+Dynamics level; MDPO justification.

---

### Chapter 4: The Adventurer Baseline Framework *(15–18 pages)*

```
  4.1 BiGAN Architecture & Joint Training Objective
  4.2 Combined Pixel & Feature Novelty Formulation B(s)
  4.3 Statistical Intrinsic Reward Normalization (Eq. 5)
  4.4 Two-Stream Policy Optimization (Extrinsic / Intrinsic GAE)
  4.5 Top-K Episodic Memory & Resettable Exploration
  4.6 Modular Codebase Architecture & Execution Flow
```

* **4.2:** $L_G(s) = \|s - G(E(s))\|_1$; $L_D(s) = \|f_D(s, E(s)) - f_D(G(E(s)), E(s))\|_1$; $B(s) = \alpha L_G(s) + (1-\alpha) L_D(s)$; transition forward-model variant $T(s_t, a_t, s_{t+1})$.
* **4.3:** $\tilde{n}(s) = (B(s) - \mu_B + \mu_{r^e}) / \sigma_B$
* **4.4:** $A_t^{\text{total}} = A_t^e(\gamma_e, \lambda) + \beta A_t^i(\gamma_i, \lambda)$
* **4.6:** Computational profiling: $G_\theta$ and $D_\omega$ forward passes account for $>60\%$ of step execution time.

---

### Chapter 5: Metric-Based Exploration in BiGAN Latent Space *(14–18 pages)*

```
  5.1 Computational & Conceptual Pitfalls of Reconstruction Novelty
  5.2 Latent State Discrepancy Formulation d_t = ||E_ψ(s_t) - E_ψ(s_{t+1})||_p
  5.3 Ensemble Reward Variance Modeling ζ(r)
  5.4 Multiplicative Integration: d_t · Scale(ζ)
  5.5 Implementation Architecture & Module Specifications
```

* **5.2:** Metric properties via BiGAN invertibility; complexity: 1 encoder pass vs. full E-G-D cycle.
* **5.3:** $K$ bootstrapped reward models; epistemic variance $\zeta$.
* **5.4:** $b_t = d_t \cdot \min(\max(\zeta(r_t), 1), M)$ (V3 formulation).

---

### Chapter 6: Normalized EME Scaling for Sparse Rewards *(14–18 pages)*

```
  6.1 The Sparse-Reward Clamping Collapse Mechanism
  6.2 Scale-Free Normalization via Exponential Moving Average (EMA)
  6.3 Graceful Degeneracy Handling & Numerical Stability Floor
  6.4 Dynamics of Reference Level Adaptation
  6.5 Preliminary Empirical Proof-of-Concept: CartPole Collapse
  6.6 Latent Metric Stabilization via Encoder Freezing
```

* **6.2:** Scale-invariance proof: $r' = cr \implies \zeta' = c^2 \zeta \implies \zeta'/\mathbb{E}[\zeta'] = \zeta/\mathbb{E}[\zeta]$.
* **6.4:** $\mathbb{E}[\zeta]_k \leftarrow m \cdot \mathbb{E}[\zeta]_{k-1} + (1-m) \cdot \overline{\zeta}_{\text{batch}}$; EMA vs. cumulative mean justification.
* **6.5:** CartPole: raw $\zeta \in \mathcal{O}(10^{-5})$–$\mathcal{O}(10^{-3})$; V3 scale locked at 1.0; V4 scale modulates 2.68–3.41.
* **6.6:** `--freeze-encoder-after-updates N`.

---

### Chapter 7: Experimental Evaluation & Benchmark Results *(22–28 pages)*

```
  7.1 Experimental Setup & Benchmark Protocols
  7.2 BiGAN State Representation & Novelty Validation
  7.3 V1–V4 Comparative Evaluation on Sparse-Reward Atari
  7.4 Comprehensive Hyperparameter Ablations
  7.5 Transition vs. State Novelty Dynamics
  7.6 Continuous Control & Robotic Manipulation Benchmarks
  7.7 Qualitative & Diagnostic Exploration Analysis
  7.8 Reproducibility, Compute Budgets, and Protocol Specifications
```

* §7.2 and §7.6 marked *(conditional)* — require external results or MuJoCo replication.

---

### Chapter 8: Discussion & Critical Analysis *(12–16 pages)*

```
  8.1 Synthesis of Empirical Findings
  8.2 Architectural Positioning vs. SOTA (RND, ICM, NGU, EME)
  8.3 Computational Efficiency & FLOP Analysis
  8.4 Theoretical & Practical Limitations
  8.5 Policy Optimizer Dynamics (PPO vs. MDPO Integration)
```

---

### Chapter 9: Conclusion & Future Outlook *(5–8 pages)*

```
  9.1 Summary of Contributions
  9.2 Methodological & Architectural Recommendations
  9.3 Future Research Directions
```

* **9.3:** Contrastive representations (SimCLR, Barlow Twins) as BiGAN alternatives; meta-learning $\beta$ scheduling; model-based RL integration (Dreamer).

---

### Appendices *(12–18 pages)*

* **A:** Detailed Hyperparameter Tables
* **B:** Extended Ablation Grid Results
* **C:** Complete Neural Network Schematics ($E_\psi, G_\theta, D_\omega$, Actor-Critic, Reward Ensemble)
* **D:** Mathematical Proofs — D.1 Scale-invariance of EMA-normalized scaling; D.2 Metric validity from BiGAN invertibility; D.3 EMA convergence under bounded variance.
* **E:** Extended Diagnostic & Trajectory Plots (t-SNE, intrinsic bonus traces)

---

## 4. Methodological & Stylistic Writing Standards

```
                              STYLE & RIGOR GUIDELINES

  ┌──────────────────────────────────────────────┬──────────────────────────────────────────────┐
  │            LINGUISTIC PRECISION              │             SCIENTIFIC RIGOR                 │
  ├──────────────────────────────────────────────┼──────────────────────────────────────────────┤
  │ • Use standard verbs: "governed by",         │ • Report mean ± 95% CI across N ≥ 3 seeds    │
  │   "relies on", "interfaces with", "denote"   │ • Prove or bound all core scaling properties │
  │ • Maintain consistent terminology:           │ • Global, sequential equation numbering      │
  │   "BiGAN", "EME", "MDPO", "PPO"              │ • Profile exact compute costs (FLOPs/time)   │
  │ • Active/passive balance with formal register│ • Zero verbatim copied text                  │
  └──────────────────────────────────────────────┴──────────────────────────────────────────────┘
```

---

## 5. Key Literature & Citation Matrix

| Primary Focus | Key Reference | Chapters |
|---------------|---------------|----------|
| Baseline Architecture | Adventurer (Liu & Liu, 2025) | 1, 2, 3, 4, 5, 7, 8 |
| Metric Exploration | EME (Wang et al., NeurIPS 2024 Spotlight) | 1, 2, 3, 5, 6, 7, 8 |
| Metric Exploration (precursor) | LIBERTY (Wang et al., NeurIPS 2023) | 2, 3, 5 |
| Policy Optimization | MDPO (Tomar et al., ICLR 2022) | 2, 3, 8 |
| Policy Optimization | PPO (Schulman et al., 2017) | 2, 3, 4, 7 |
| Policy Optimization | TRPO (Schulman et al., 2015a) | 2, 3 |
| Adversarial Learning | BiGAN (Donahue et al., ICLR 2017) | 2, 4, 5, 8 |
| Adversarial Learning | ALI (Dumoulin et al., ICLR 2017) | 2, 3 |
| Novelty Benchmarks | RND (Burda et al., ICLR 2019) | 1, 2, 3, 4, 7, 8 |
| Novelty Benchmarks | ICM (Pathak et al., ICML 2017) | 2, 3, 8 |
| Novelty Benchmarks | GAEX (Hong et al., DAI 2019) | 2, 3, 8 |
| Memory-Based RL | NGU (Badia et al., ICLR 2020a) | 2, 3, 8 |
| Memory-Based RL | Agent57 (Badia et al., 2020b) | 2, 3, 8 |
| Memory-Based RL | Go-Explore (Ecoffet et al., Nature 2021) | 2, 3 |
| Metric Theory | Bisimulation (Ferns et al., UAI 2004) | 2, 3 |
| Metric Theory | DBC (Zhang et al., NeurIPS 2020) | 2, 3 |
| Adaptive Intrinsic Scaling | CIM (Zheng et al., IJCAI 2024) | 2, 3, 6 |
| Intrinsic Reward Taxonomy | Kayal et al. (Neural Comput & Applic 2025) | 2, 3, 6 |
| Exploration Surveys | Yang et al. (2021); Amin et al. (2021); Ladosz et al. (2022) | 2, 3 |

---

## 6. Pre-Writing Evidence & Verification Tracking Matrix

| Priority | Required Item | Chapters | Status | Action Required |
|----------|---------------|----------|--------|-----------------|
| 🔴 Critical | Benchmark Runs: V1–V4 on Sparse Atari (*Montezuma*, *Gravitar*, *Solaris*) × 3 seeds | Ch. 7 | ❌ Pending | Run `run_metric_eme_comparison.sh` on GPU cluster |
| 🔴 Critical | Hyperparameter Ablations ($\alpha, \beta, K, M, L_1/L_2$, frozen encoder) | Ch. 7 | ❌ Pending | Execute automated sweep scripts |
| 🟡 Important | Formal Proof: Scale-invariance of $\zeta / \mathbb{E}[\zeta]$ | Ch. 6, App. D | ⚠️ Outlined | Draft lemma: $r \to cr \implies \zeta \to c^2\zeta \implies$ ratio invariant |
| 🟡 Important | Metric Space Verification: BiGAN encoder induced latent metric | Ch. 5, App. D | ⚠️ Outlined | Prove metric axioms via BiGAN invertibility ($E = G^{-1}$) |
| 🟡 Important | Computational Profiling: FLOPs & wall-clock (V1 vs. V4) | Ch. 8 | ❌ Needed | PyTorch profiler on single rollout worker |
| 🟢 Desirable | Continuous Control Benchmarks (*FetchPickAndPlace*, *HandManipulateBlock*) | Ch. 7 | ⚠️ Code Ready | Execute MuJoCo validation |
| ✅ Verified | EME Theoretical Framework (Theorems 1–2, Props 1–3, Def. 2) | Ch. 2, 3, 5 | ✅ Complete | Extracted from NeurIPS 2024 (OpenReview: QpKWFLtZKi) |
| ✅ Verified | Core Implementations (`state_discrepancy.py`, `ensemble_scaling.py`, `intrinsic_reward.py`) | Ch. 4, 5, 6 | ✅ Complete | Verified in codebase with passing tests |
| ✅ Verified | MDPO (Tomar et al., 2022) | Ch. 2 | ✅ Complete | Extracted from arXiv:2005.09814 |
| ✅ Verified | LIBERTY (Wang et al., 2023) | Ch. 2, 3 | ✅ Complete | Extracted from NeurIPS 2023 PDF |
| ✅ Verified | Exploration Surveys (Yang, Amin, Ladosz, Kayal) | Ch. 2, 3 | ✅ Complete | Extracted from arXiv / Springer |
| ✅ Verified | Ferns et al. (2004) Bisimulation | Ch. 2 | ✅ Complete | Extracted from arXiv:1207.4114 |
| ✅ Verified | CIM (Zheng et al., 2024) | Ch. 3, 6 | ✅ Complete | Extracted from arXiv:2407.09247 |

---

## 7. Inter-Chapter Dependency Graph & Optimal Writing Workflow

```
                    INTER-CHAPTER DEPENDENCY GRAPH

                         ┌───────────────────┐
                         │   Chapter 2       │
                         │  (Background)     │
                         └─────────┬─────────┘
                                   │
              ┌────────────────────┼────────────────────┐
              ▼                    ▼                    ▼
    ┌───────────────────┐┌───────────────────┐┌───────────────────┐
    │   Chapter 3       ││   Chapter 4       ││   Chapter 5       │
    │  (Related Work)   ││  (Adventurer)     ││ (Latent Metric)   │
    └───────────────────┘└─────────┬─────────┘└─────────┬─────────┘
                                   │                    │
                                   └──────────┬─────────┘
                                              ▼
                                    ┌───────────────────┐
                                    │   Chapter 6       │
                                    │ (Normalized EME)  │
                                    └─────────┬─────────┘
                                              │
                                              ▼
                                    ┌───────────────────┐
                                    │   Chapter 7       │
                                    │  (Experiments)    │
                                    └─────────┬─────────┘
                                              │
                                   ┌──────────┴──────────┐
                                   ▼                     ▼
                         ┌───────────────────┐ ┌───────────────────┐
                         │   Chapter 8       │ │   Chapter 1       │
                         │  (Discussion)     │ │ (Introduction)    │
                         └─────────┬─────────┘ └─────────┬─────────┘
                                   │                     │
                                   └──────────┬──────────┘
                                              ▼
                                    ┌───────────────────┐
                                    │   Chapter 9       │
                                    │  (Conclusion)     │
                                    └───────────────────┘
```

### Optimal Phased Writing Sequence

1. **Phase 1 (Foundations):** Chapter 2 (Background) → Establish all formal definitions, MDP notations, BiGAN inversion theorems, and EME bounds.
2. **Phase 2 (Survey & Baseline):** Chapter 3 (Related Work) and Chapter 4 (Adventurer Framework) → Survey the exploration landscape and formalize the baseline.
3. **Phase 3 (Methodology):** Chapter 5 (Latent Metric) and Chapter 6 (Normalized Scaling) → Core theoretical and algorithmic contributions.
4. **Phase 4 (Empirical Execution & Analysis):** Execute experiments → Chapter 7 (Experiments) and Chapter 8 (Discussion).
5. **Phase 5 (Synthesis & Front Matter):** Chapter 1 (Introduction), Chapter 9 (Conclusion), and Appendices A–E.

---

## 8. Unified Mathematical Notation Index

| Symbol | Meaning | First Appearance |
|--------|---------|-----------------|
| $\mathcal{S}, \mathcal{A}, \mathcal{R}$ | State, action, and reward spaces | Ch. 2 |
| $s_t, a_t, r_t^{\text{ext}}$ | State, action, and extrinsic reward at time $t$ | Ch. 2 |
| $\pi(a|s), \pi_k$ | Policy distribution, policy at iteration $k$ | Ch. 2 |
| $V^\pi(s), Q^\pi(s,a)$ | State-value and action-value functions | Ch. 2 |
| $A^\pi(s,a)$ | Advantage function | Ch. 2 |
| $D_{\text{KL}}(\cdot \| \cdot)$ | Kullback-Leibler divergence | Ch. 2 |
| $t_k$ | MDPO temperature / step-size parameter | Ch. 2 |
| $r_t^{\text{int}}, r_t^{\text{total}}$ | Intrinsic reward and total composite reward | Ch. 2 |
| $N(s), \tilde{N}(s)$ | Exact and pseudo-visit counts | Ch. 2 |
| $d_E(s_i, s_j)$ | EME bisimulation metric distance | Ch. 2 |
| $E_\psi$ | BiGAN encoder with parameters $\psi$ | Ch. 4 |
| $G_\theta$ | BiGAN generator with parameters $\theta$ | Ch. 4 |
| $D_\omega$ | BiGAN joint discriminator with parameters $\omega$ | Ch. 4 |
| $L_G(s)$ | Pixel reconstruction loss $\|s - G(E(s))\|_1$ | Ch. 4 |
| $L_D(s)$ | Feature-matching loss $\|f_D(s, E(s)) - f_D(G(E(s)), E(s))\|_1$ | Ch. 4 |
| $B(s)$ | Composite novelty score $\alpha L_G(s) + (1-\alpha)L_D(s)$ | Ch. 4 |
| $\tilde{n}(s)$ | Normalized novelty (Adventurer Eq. 5) | Ch. 4 |
| $\mu_B, \sigma_B, \mu_{r^e}$ | Novelty mean/std, extrinsic reward mean | Ch. 4 |
| $\gamma_e, \gamma_i, \lambda$ | Extrinsic/intrinsic discount, GAE decay | Ch. 4 |
| $\beta$ | Intrinsic advantage weighting coefficient | Ch. 4 |
| $f_\phi$ | Latent forward dynamics model | Ch. 4 |
| $d_t$ | Latent state discrepancy $\|E_\psi(s_t) - E_\psi(s_{t+1})\|_p$ | Ch. 5 |
| $p$ | Norm order for latent discrepancy ($p \in \{1, 2\}$) | Ch. 5 |
| $\hat{r}_k(s, a)$ | Reward prediction of ensemble model $k$ | Ch. 5 |
| $\zeta(r)$ | Epistemic reward variance across $K$-model ensemble | Ch. 5 |
| $M$ | Maximum reward scaling factor (upper clamp) | Ch. 5 |
| $K$ | Number of bootstrapped models in reward ensemble | Ch. 5 |
| $b_t$ | Full metric exploration bonus | Ch. 5 |
| $\mathbb{E}[\zeta]$ | EMA reference variance level | Ch. 6 |
| $m$ | EMA momentum coefficient (default: 0.99) | Ch. 6 |
| $\zeta_\epsilon$ | Numerical floor for ensemble degeneracy detection | Ch. 6 |
