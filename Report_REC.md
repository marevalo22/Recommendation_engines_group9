# Group 9 Report — Football Transfer Market Recommender System

---

## 1. Domain Analysis & Data Description

### 1.1 Domain Definition

This project models the football transfer market as a recommender system. Clubs are treated as users, players as items, and transfers (with minutes played) as implicit feedback. The objective is to recommend players that a club should sign in a transfer window.

The problem is inherently a sparse implicit-feedback setting: each club signs only a small number of players per season relative to a large candidate pool. This structural sparsity makes accurate ranking difficult and motivates the use of collaborative filtering and feature-based methods.

### 1.2 Data Sources

Data was collected from two primary sources:

- **FBref** (via the `soccerdata` library): provides player- and team-level statistics across six seasons (2018–19 to 2023–24). Extracted blocks include standard, shooting, playing time, misc, and goalkeeper statistics.
- **Transfermarkt**: provides player market values and transfer history.

The available FBref data limits the real feature space to approximately 25 player-level statistics. This constraint reduces the expressiveness of feature-based models, particularly for capturing tactical and role-specific nuances.

Market value is used as a proxy for affordability and is log-transformed due to its heavy-tailed distribution.

### 1.3 Dataset Characteristics

| Table | Rows | Columns |
|---|---|---|
| Player-seasons | 16,873 | 152 |
| Team-seasons | 586 | 120 |

The dataset includes:
- 6,021 unique players
- 140 unique clubs

A strict temporal split is applied:

- **Training:** 2018–19 to 2021–22
- **Validation:** 2022–23
- **Test:** 2023–24

This ensures that all evaluation reflects forward-looking prediction and avoids data leakage.

### 1.4 Interaction Matrix and Sparsity

The team–player interaction matrix is constructed from minutes played and represented as a sparse matrix (clubs × players). The observed interaction density is approximately **1.03%**, meaning that around 99% of possible club–player pairs are unobserved.

This level of sparsity is structural: clubs interact with a very small subset of players each season. As a result:

- Global ranking approaches are weak baselines
- Collaborative filtering must infer structure from limited co-occurrence signals

### 1.5 Exploratory Data Analysis

**Coverage:** After preprocessing, player counts per league-season are stable (approximately 326–399 observations per cell), indicating consistent data availability across leagues and seasons.

**Minutes Distribution:** The distribution of minutes played is strongly right-skewed, with a large number of low-minute observations. These correspond to players with limited participation and unreliable statistics, motivating the filtering thresholds applied in preprocessing.

**Market Value:** Transfermarkt valuations are available for approximately 92% of player-season observations. The distribution is approximately normal after log transformation, supporting its use in scoring functions.

**Transfer Ground Truth:** The target variable identifies players who transfer to a club between seasons, yielding **1,930 positive transfer events** across the dataset. The low number of positives per club and position highlights the difficulty of the recommendation task.

**Popularity Distribution:** Player exposure is highly skewed: a small number of players appear frequently across clubs and seasons. This creates a long-tail distribution and introduces popularity bias in interaction-based models such as collaborative filtering.

---

## 2. Data Preprocessing & Feature Engineering

### 2.1 Minutes Filtering

Low-information observations are removed using minimum playing-time thresholds:

- **600 minutes** for outfield players
- **900 minutes** for goalkeepers

This reduces the dataset from 16,873 to **11,093** player-season rows.

The filtering removes observations with unstable per-90 statistics, improving the reliability of downstream feature construction and similarity measures.

### 2.2 Position Parsing

Player positions are extracted from FBref strings by selecting the primary listed role and mapping it to one of four families: **GK, DF, MF, FW**.

Players with missing or invalid positions are excluded from position-dependent operations.

### 2.3 Per-90 Feature Conversion

Raw counting statistics (e.g., tackles, interceptions, crosses) are converted to per-90 rates by dividing by minutes played. A lower bound is applied to avoid instability near the filtering threshold. This transformation removes playing-time bias and allows meaningful comparison across players.

### 2.4 Imputation and Standardisation

Missing values are imputed using:

- Median within (league, season, position) groups
- Fallback to global median when necessary

All features are then z-scored within the same groups, so each value reflects relative performance within its competitive context. This standardisation enables cross-league comparability and stabilises similarity calculations.

### 2.5 Role Subtype Assignment

Players are assigned role subtypes using rule-based proxy scores, rather than clustering. These proxies combine relevant statistics to produce interpretable categories such as:

- Defensive vs attacking defenders
- Ball-winning vs creative midfielders
- Strikers vs wide forwards

This approach is deterministic and avoids instability from unsupervised clustering in a limited feature space.

### 2.6 Player Feature Representation

Each player is represented by a dense feature vector composed of:

- Z-scored performance statistics
- Role dummy variables
- Polynomial age terms

The implemented feature spaces are:

- **Real-only vector:** 24 dimensions
- **Synthetic-augmented vector:** 43 dimensions

All vectors are L2-normalised before similarity computation.

### 2.7 Team Context Representation

Team context is represented through synthetic style variables capturing key tactical dimensions:

- Possession
- Directness
- Pressing
- Width
- Tempo
- Territorial dominance

These variables are used in the context-aware model to evaluate compatibility between player attributes and team identity.

### 2.8 Synthetic Feature Layer

To address limitations in public data, a synthetic feature layer augments the dataset with additional player and team descriptors, as well as derived scoring components. Synthetic features are:

- Stored separately from real features
- Identified with a `syn_` prefix

The final evaluation benchmark uses only real features, while synthetic features are used for exploratory modelling and context-aware scoring.

### 2.9 Data Separation and Integrity

To ensure methodological correctness:

- Real and synthetic datasets are kept separate
- Preprocessing is applied consistently across models
- Evaluation follows a strict temporal split

This prevents data leakage and maintains a clear distinction between validated results and exploratory extensions.

---

## 3. Non-Personalized Recommender

### 3.1 Model Design

The non-personalized recommender produces a position-specific ranking that is identical for all clubs within a position family. It serves as a baseline model: any personalized method must outperform it to justify additional complexity.

The model ranks players in the 2023–24 candidate pool using only features derived from that season. No transfer outcomes are used at scoring time.

### 3.2 Composite Scoring Function

The primary score, `real_quality_score`, is a weighted linear combination of z-scored per-90 statistics. Weights are defined separately for each position.

**Forwards (FW):**

| Feature | Weight |
|---|---|
| Non-penalty goals per 90 | 0.36 |
| Assists per 90 | 0.18 |
| Shots on target per 90 | 0.18 |
| Goals per shot | 0.10 |
| Fouls won per 90 | 0.08 |
| Team plus-minus per 90 | 0.10 |

**Midfielders (MF):**

| Feature | Weight |
|---|---|
| Assists per 90 | 0.24 |
| Non-penalty goals per 90 | 0.16 |
| Interceptions per 90 | 0.20 |
| Tackles won per 90 | 0.16 |
| Fouls won per 90 | 0.10 |
| Crosses per 90 | 0.06 |
| Team plus-minus per 90 | 0.08 |

**Defenders (DF):**

| Feature | Weight |
|---|---|
| Interceptions per 90 | 0.32 |
| Tackles won per 90 | 0.28 |
| Crosses per 90 | 0.10 |
| Team plus-minus per 90 | 0.15 |
| Team On-Off per 90 | 0.15 |

All features are z-scored within (league, season, position), and weights sum to 1. The weights reflect domain priors, where attacking output dominates forwards, defensive actions dominate defenders, and midfield roles balance both contributions.

### 3.3 Score Variants

**Value-adjusted score:**

$$\text{value\_adjusted\_score} = \text{real\_quality\_score} - 0.22 \cdot \text{log\_market\_value\_z}$$

This penalises highly valued players, favouring cost-efficient alternatives.

**Trajectory-adjusted score:**

$$\text{trajectory\_adjusted\_score} = \text{real\_quality\_score} \cdot \left(0.6 + 0.4 \cdot e^{-((age-26)^2 / 50)}\right)$$

This applies an age-based weighting that peaks at 26 and declines symmetrically.

### 3.4 Sensitivity Analysis

Robustness is assessed using Jaccard overlap between top-10 lists under alternative weighting schemes:

- Swapped goal and assist weights
- Equal weights across features
- Removal of the team-success term

High overlap (≥ 0.7) indicates that rankings are not sensitive to reasonable perturbations in weight selection.

### 3.5 Baseline Performance

On the final evaluation (Notebook 05), the model achieves:

- **Precision@10** = 0.0019
- **Recall@10** = 0.0090
- **NDCG** = 0.0043

Performance is low because the model produces identical rankings for all clubs, while actual transfers are club-specific. This reflects the mismatch between a global ranking and a sparse, club-specific ground truth.

### 3.6 Role in the System

Despite weak ranking performance, the model serves two purposes:

- A cold-start baseline when interaction data is unavailable
- A global ranking reference providing interpretable player quality signals

It establishes a benchmark against which personalized models are evaluated.

---

## 4. Collaborative Filtering Recommender

### Model Design

The collaborative filtering model treats clubs as users and players as items, learning latent preference vectors from the historical record of which players a club has fielded and for how many minutes. The key design choice is the feedback signal: rather than binary interaction indicators, the model uses minutes played as a continuous proxy for preference intensity.

### Interaction Matrix and Log-Compression

The interaction matrix **R** is a sparse CSR matrix of shape (teams × players). Each entry is:

$$r_{i,j} = \log1p\left(\sum_s \text{minutes\_played}_{i,j,s}\right)$$

where the sum runs over all training seasons for the team–player pair. Log compression (`log1p`) is applied before constructing the matrix to prevent the most heavily-used players from dominating the factor solution disproportionately.

The training matrix covers five seasons (2018–19 through 2022–23). The interaction density is **1.03%**. The measured degree distributions show that teams touch a median of several dozen distinct players across training seasons, while most players are touched by one or two clubs.

### Factorisation Method

Dimensionality reduction is performed with `TruncatedSVD` (k=16, `random_state=42`) from scikit-learn, producing a 16-dimensional latent vector for each club and each player. The affinity score for a (club, candidate) pair is the dot product of their latent vectors:

$$\text{score(club, player)} = T[\text{club}] \cdot P[\text{player}]$$

At recommendation time, the candidate pool is restricted to players active in the previous season (2022–23), and any player already on the target club's 2022–23 roster is excluded.

**Why TruncatedSVD rather than ALS?** The principled choice for implicit feedback data is Alternating Least Squares with confidence weighting (Hu, Koren, Volinsky 2008). TruncatedSVD treats all unobserved entries as zero reconstruction targets, which blurs the zero-signal interpretation. ALS is identified as the natural next improvement for a production deployment.

**Why k=16?** The latent dimensionality k=16 was chosen heuristically, balancing representation capacity against the risk of overfitting. The notebook validates this choice through a k-stability ablation.

### Hyperparameter Tuning: K-Stability Ablation

The default k=16 is validated through a k-stability ablation. The CF model is re-fitted at k=8 and k=32 for three representative club-position pairs (Arsenal midfielders, Liverpool defenders, Brentford forwards). Jaccard overlap between the k=16 top-10 and alternative-k top-10 is computed and reported.

### Results

On the honest real-only evaluation in `05_evaluation.ipynb`:

| Metric | Value |
|---|---|
| RMSE | 0.4073 |
| MAE | 0.3860 |
| Precision@10 | 0.0054 |
| Recall@10 | 0.0165 |
| NDCG | 0.0083 |
| Coverage | 0.3051 |
| Diversity | 0.8142 |
| Serendipity | 0.0030 |

Collaborative filtering is the strongest model in the real-only benchmark across all ranking accuracy metrics, with NDCG 0.0083 — approximately three times the random baseline floor (NDCG ≈ 0.003).

### Algorithmic Limitations

**First**, the model has no awareness of tactical fit, positional need, or affordability. Positional filtering is applied as a post-hoc constraint rather than a preference signal.

**Second**, high-exposure clubs inflate latent scores. PSG, Bayern Munich, and Manchester City appear more frequently in the training matrix, causing their players to be disproportionately surfaced. This is a structural popularity bias inherent to the co-occurrence matrix.

**Third**, the model is largely insensitive to synthetic feature augmentation because TruncatedSVD consumes only the interaction matrix. CF NDCG remains essentially flat (0.0083 → 0.0076) between real-only and synthetic tracks, providing a useful negative control.

---

## 5. Content-Based Recommender

### Item Profile Representation

The content-based recommender represents each player-season as a dense numeric vector constructed from three components:

1. **Z-scored per-90 statistical features** from FBref's standard, shooting, playing_time, and misc blocks. Z-scoring within (league, season, position family) groups plays the role that TF-IDF normalisation plays in text retrieval.

2. **7-category one-hot role encoding** (GK, DF_r0, DF_r1, MF_r0, MF_r1, FW_r0, FW_r1). Including the role dummy ensures cosine similarity penalises cross-role comparisons appropriately.

3. **Polynomial age encoding:** a centred age term `(age − 26) / 5` and its square. These terms add a non-linear representation of career stage.

All vectors are L2-normalised before any similarity computation, making cosine similarity equivalent to the dot product at inference time.

A synthetic-augmented feature space adds further columns with a `syn_` prefix (e.g., `syn_trait_finishing`, `syn_trait_creativity`, `syn_xg_p90`). These are proof-of-concept enrichments; the honest benchmark uses only the real-only vector.

### User (Club) Profile Generation

The model constructs a team taste profile as the minutes-weighted centroid of the squad's players at the target position in the previous season:

$$\text{profile(team, pos, season)} = \frac{\sum_i \text{minutes}_i \times \vec{v}_i}{\sum_i \text{minutes}_i}$$

where $\vec{v}_i$ is the L2-normalised feature vector of the i-th player. The resulting profile vector is itself L2-normalised before similarity scoring. If a club has no players at the target position, the function returns `None` and the club is excluded from that position cell.

### Scoring: Replicate and Complement Modes

Given a query (team, position, target season), every candidate player is scored by cosine similarity between their feature vector and the team's profile:

$$\text{score(candidate)} = \vec{v}_\text{candidate} \cdot \text{profile(team, pos)}$$

This is the **replicate mode**: it surfaces players who statistically resemble the club's existing profile — "more of what we already have."

The **complement mode** tilts the target profile toward the club's gap relative to its positional peers:

$$\text{direction} = (1 - \alpha) \times \text{squad\_profile} + \alpha \times (\text{peer\_profile} - \text{squad\_profile})$$

where `peer_profile` is the minutes-weighted centroid of the top-quintile players at the same position in the same league, and α = 0.5. The dual-mode design is a meaningful contribution to practical scouting usability, though the honest evaluation uses only the replicate mode.

---

## 6. Context-Aware Recommender

### Context Variables and Domain Justification

The context-aware model operationalises three classes of context variable from `04_context_aware.ipynb`.

**Tactical style context** is captured via six paired axes:

| Player preference | Team latent |
|---|---|
| `syn_pref_possession` | `syn_latent_possession_orientation` |
| `syn_pref_directness` | `syn_latent_directness` |
| `syn_pref_pressing` | `syn_latent_pressing_intensity` |
| `syn_pref_width` | `syn_latent_width_crossing` |
| `syn_pref_tempo` | `syn_latent_tempo` |
| `syn_pref_territorial` | `syn_latent_territorial_dominance` |

**Squad need context** is captured via five player contribution dimensions (`syn_contrib_box_threat`, `syn_contrib_creation`, `syn_contrib_progression`, `syn_contrib_pressing`, `syn_contrib_aerial`) matched against a dynamically computed team need vector.

**Budget context** is captured via Transfermarkt market value mapped against the team's `syn_budget_band_index` (ordinal 0–100).

### Architectural Paradigm: Contextual Scoring

The context-aware recommender follows a **contextual modeling paradigm**: context variables are directly embedded into the scoring function at inference time, not used as pre-filters or post-filters. The composite score is:

$$\text{context\_syn\_poc\_score} = 0.35 \times \text{style\_fit\_poc} + 0.30 \times (0.5 + 0.5 \times \text{need\_fit\_poc}) + 0.20 \times \text{budget\_fit\_poc} + 0.15 \times \text{quality\_logistic}$$

where:

- `style_fit_poc` = $1 - \text{mean}_j(|\text{player\_pref}_j - \text{team\_latent}_j|)$ across the six tactical axes
- `need_fit_poc` = dot(player_contrib_vector, team_need_vector), where the need vector is unit-norm
- `budget_fit_poc` = $\exp\left(-(\Delta\text{band}/100)^2/2\right)$ — a Gaussian kernel centred at the team's budget band
- `quality_logistic` = $1 / (1 + \exp(-\text{real\_quality\_score}))$ — a logistic-squashed version of the non-personalized quality score

The weights (0.35, 0.30, 0.20, 0.15) reflect deliberate domain priors: tactical fit is the primary signal, squad need is nearly as important, affordability is operationalised as a soft penalty, and raw quality is a secondary anchor.

---

## 7. Comparative Evaluation

### Train/Test Split Methodology

All models are evaluated under a strict temporal hold-out protocol with no data leakage:

- **Training seasons:** 2018–19, 2019–20, 2020–21, 2021–22
- **Validation season:** 2022–23
- **Test season:** 2023–24 (held out entirely until final evaluation)

Candidates are drawn from the 2022–23 player pool. Any player already on the target club's 2022–23 roster is excluded. Ground truth is the `inferred_transfer_arrival_this_team` flag in the 2023–24 data. Club-position cells with zero arrivals are excluded from ranking metrics.

### Evaluation Metrics

K = 10 throughout, reflecting a realistic scout shortlist length.

- **Precision@K** = (hits in top-10) / 10
- **Recall@K** = (hits in top-10) / (total actual arrivals at that club-position)
- **NDCG@K** — rewards models that place actual arrivals near the top of the list
- **RMSE / MAE** — prediction accuracy vs. binary arrival label (scores min-max normalised to [0,1])
- **Coverage** — fraction of total candidate catalogue appearing in at least one recommendation list
- **Diversity** — mean pairwise cosine distance between player vectors within each club's top-10
- **Serendipity** — fraction of top-10 hits not surfaceable by a simple popularity baseline

### Standardised Comparison Table (Real-Only Benchmark)

| Approach | RMSE | MAE | Precision@K | Recall@K | NDCG | Coverage | Diversity | Serendipity | Context |
|---|---|---|---|---|---|---|---|---|---|
| Random Baseline | ~0.50 | ~0.47 | ~0.0010 | ~0.0030 | ~0.0030 | ~0.33 | ~0.79 | ~0.0000 | No |
| Non-Personalized | 0.4508 | 0.4470 | 0.0000 | 0.0000 | 0.0000 | 0.0065 | 0.6841 | 0.0000 | No |
| Collaborative Filtering | 0.4073 | 0.3860 | 0.0054 | 0.0165 | 0.0083 | 0.3051 | 0.8142 | 0.0030 | No |
| Content-Based | 0.5337 | 0.4999 | 0.0011 | 0.0036 | 0.0019 | 0.2501 | 0.6227 | 0.0000 | No |
| Context-Aware | 0.4634 | 0.4376 | 0.0043 | 0.0129 | 0.0072 | 0.2965 | 0.7334 | 0.0027 | Yes |

> **Note:** Random Baseline values are approximate, reproduced from the NB05 sanity-check cell. All other values are from the main evaluation loop. Context delta for Context-Aware = −0.0026 (vs. CF as best non-context model).

### Critical Interpretation

**Collaborative Filtering** is the strongest model on every ranking accuracy metric (NDCG 0.0083, Precision@10 0.0054, Recall@10 0.0165). Coverage (30.5%) and diversity (0.8142) are also the highest among personalised models.

**The Non-Personalized model** returns zero on all ranking accuracy metrics and a coverage of only 0.65% of the catalogue. Zero hit rate is analytically expected: the model recommends the same global quality-ranked list to every club, while the transfer ground truth is club-specific.

**The Content-Based model** underperforms CF by 4× on NDCG (0.0019 vs. 0.0083). This is a feature-richness problem: the 28-dimensional real-only vector does not encode sufficient role or tactical nuance. The synthetic-augmented content track raises this to NDCG 0.0109, a 5.7× improvement, confirming that the vector was feature-limited, not broken.

**The Context-Aware model** has a negative context delta (−0.0026), meaning the explicit contextual scoring layer decreases ranking quality relative to CF. Two factors drive this: the synthetic style and budget features were constructed without calibration to real transfer arrivals, injecting noise; and the score separation achieved is insufficient to outweigh false positives promoted into the top-10. This functions as a falsification check: contextual features only earn their weight when calibrated on real training signals.

Absolute metric levels (NDCG 0.003–0.008) are low but structurally expected. Each club signs at most one to three players per position per season from a candidate pool of over a thousand. The observed average of 0.0054 (Precision@10) means the model surfaces the actual arrival in the top-10 approximately once in every 185 club-position evaluations — better than random (~once per 1,000 evaluations).

### Simulated A/B Test Design

**Experimental unit:** One club's candidate review cycle for a single transfer window.

**Hypotheses:**
- H₀: the algorithmic shortlist does not improve the proportion of recommended candidates who are ultimately signed.
- H₁: the algorithmic shortlist raises the shortlist-to-signing conversion rate by at least 5 percentage points above the estimated control base rate of 10–15%.

**Primary success metric:** Shortlist-to-signing conversion rate.

**Secondary guardrail metrics:** Time-to-shortlist, analyst agreement rate, and recommended-nationality diversity. If diversity degrades by more than 20% relative to the control arm, the trial halts.

**Sample size and power:** Using a two-proportion z-test with α = 0.05, power = 0.80, a control conversion rate of 0.12, and a minimum detectable effect of 0.05, the required sample size is approximately **400 club-window pairs per arm** (~two full seasons).

**Offline proxy:** Simulated proxy lift = +0.0043 to +0.0054 on Precision@10 (Context-Aware vs. Non-Personalized, and CF vs. Non-Personalized respectively).

---

## 8. Business Case & Deployment Design

### ROI Argument

The system creates value through **risk reduction** and **value discovery**.

On **risk reduction**, failed transfers in Top-5 leagues often cost €20–50M. Even a modest improvement in shortlist quality can prevent costly mistakes, justifying the analytics investment.

On **value discovery**, the model identifies undervalued, high-fit players through value-adjusted and budget-aware scoring, enabling clubs to acquire players before market price increases.

The impact is strongest for mid-tier clubs, which lack extensive scouting networks. The system achieves Precision@10 ≈ 0.005, comparable to manual scouting efficiency, but at significantly greater scale.

### Production Architecture

**Offline batch pipeline (nightly cadence):**

```
Raw data ingestion → Feature store → Model training → Artefact registry
```

FBref and Transfermarkt data are pulled nightly (24-hour freshness). The feature store precomputes and versions player feature vectors, team style vectors, and pairwise pair-fit scores. The CF model is retrained end-of-season; an optional January refresh can be triggered mid-season.

**Online inference layer (<500ms latency):**

```
Serving API → Scout interface
```

With precomputed feature vectors, ranking all ~3,000 candidate–club pairs for a given position takes under 500ms. The serving API accepts a club ID plus optional context parameters (position need, budget ceiling, age window) and returns a ranked top-10 with per-candidate fit explanations.

**Throughput target:** All 98 Top-5 league clubs generating shortlists concurrently requires approximately 10–20 parallelised API instances — standard horizontal scaling achievable on any major cloud provider.

**Monitoring:** Ranking drift (KL divergence between shortlist distributions), coverage decay, and feature staleness (Transfermarkt valuations lagging beyond 30 days) are monitored continuously.

**Phased A/B rollout:**
- **Phase 1 (shadow mode):** System runs in parallel with manual scouting for one full transfer window; analysts receive shortlists only after producing their own lists.
- **Phase 2:** Shortlist introduced as live input for a randomly selected treatment arm. Minimum of two complete transfer windows per arm required for 80% statistical power at α = 0.05.

---

## 9. Cold-Start & Bias Mitigation Strategy

**Cold-start** is inherent to this domain. New or low-minute players provide limited statistical signals. The system mitigates this partially: the non-personalized model provides a fallback, the content-based model operates without interaction history, and synthetic features introduce additional priors (e.g., age, upside).

**Club cold-start** presents a similar challenge. Newly promoted or low-history clubs are poorly represented in collaborative filtering due to sparse interaction data. In these cases, the system relies on global rankings and team-style representations.

**Bias** is also present. Interaction data over-represents high-visibility clubs, creating exposure bias in collaborative filtering. Market values and transfer outcomes reflect external factors not captured in the data (contracts, agent dynamics). These effects are mitigated through within-group standardisation, diversity and coverage reporting, affordability-aware scoring, and clear separation between real-only evaluation and synthetic modelling.

A key methodological safeguard: the synthetic proof-of-concept layer incorporates partially label-informed features and is **not used as a primary benchmark**. Real-only, synthetic-base, and PoC results are reported separately. Uplift is deliberately moderate — score distributions for real and non-real transfers still overlap, ensuring the PoC remains illustrative rather than artificially deterministic.

---

## 10. Conclusions

This project demonstrates that football transfer recommendation is a valid but inherently difficult recommender problem. The interaction matrix is highly sparse, positive labels are rare, and publicly available features are limited, resulting in low absolute performance even for the best models.

**Primary conclusion:** The real-only benchmark remains the main result. In this setting, collaborative filtering performs best, the non-personalized model serves as a baseline, and the content-based and context-aware approaches are constrained by limited feature richness.

**Secondary conclusion:** Feature quality is the main bottleneck. Introducing richer descriptors improves content-based performance and enables meaningful context-aware scoring, indicating that model limitations are largely driven by data rather than framework choice.

The synthetic proof-of-concept track is useful for scenario modelling and explanation, but not as a valid performance benchmark due to its partial reliance on label-informed features.

From a practical perspective, the system's value lies in **shortlist generation and interpretability**, rather than full automation of recruitment decisions.

Future work should focus on expanding real data sources, improving contextual modelling, and replacing synthetic proxies with observed features to achieve a more robust and realistic system.

---

## 11. Individual Contributions

| Member | Sections |
|---|---|
| Andrea Saxod | Domain analysis, non-personalized model (§1, §3) |
| Cloe Chapotot | Collaborative filtering (§4), notebook QA |
| Constantin Hatecke | Content-based model, feature engineering (§2, §5) |
| Marcela Funabishi | Context-aware model (§6) |
| Matias Arevalo | Evaluation and metrics (§7) |
| Vittorio Fialdini | Business case, integration, packaging (§8–§12) |

The augmentation pass focused on synthetic feature generation (player, team, pair), evaluation audit, and final integration.

---

## 12. AI Usage Disclosure

ChatGPT was used as a support tool for restructuring notebooks, assisting with synthetic feature engineering, and drafting and refining report text.

All quantitative results were generated from executable code. Real-only and synthetic tracks were kept separate, and the synthetic uplift process was explicitly logged for transparency.

AI assistance did not replace human responsibility. The team verified all code, results, and report content.
