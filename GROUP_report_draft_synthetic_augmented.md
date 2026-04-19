# Football Player Recommender — Synthetic-Augmented Draft Report

## 1. Domain Analysis & Data Description

This project frames football transfer targeting as a recommender-systems problem where the **users are clubs** and the **items are players**. That framing already existed in the original project materials and remains the correct one after inspection of the draft notebook, executed notebook, README, assignment brief, and master plan. The project objective is not to predict who the best footballer is in the abstract, but to rank a shortlist of players who fit a particular club's tactical identity, positional needs, squad profile, and value constraints.

The current project folder already contained a credible first draft of that pipeline. The key assets were a clean notebook, an executed notebook with figures and outputs, a notebook-construction script, and four consolidated data files. The draft had already implemented all four required recommender families: non-personalised, collaborative filtering, content-based, and context-aware. It also already used a sensible football evaluation framing based on historical transfer outcomes rather than explicit ratings.

After inspection, the existing dataset spine can be summarised as follows. The player master file contains **16,873 player-season rows across 152 columns**. The team master file contains **586 team-season rows across 120 columns**. The working scope is narrower than the original master plan: the current dataset covers the **Top-5 European leagues only**, not the broader nine-league scope originally envisioned. The observed seasons run from **2018-19 through 2023-24**, with about **6,021 unique players** in the consolidated data and market-value coverage of roughly **92%**. After the draft notebook's minutes filters, the evaluation set contains **94 validation teams / 322 signings** in 2022-23 and **93 test teams / 345 signings** in 2023-24. This is a realistic but difficult setting: each club signs very few players relative to the candidate pool, so absolute ranking metrics remain low.

The original notebook had several genuine strengths. First, it already respected a time-based split rather than using a leaky random split. Second, it had sensible football-specific ground truth construction based on actual arrivals and target-season minutes. Third, it used role clustering to go beyond Transfermarkt-style coarse position labels. Fourth, it included beyond-accuracy metrics such as coverage, diversity, and serendipity, which is exactly what the course brief expects from the final evaluation.

At the same time, the current project was also missing some of the richer context that makes football scouting recommender systems genuinely interesting. The content-based representation was limited by the available consolidated public features. The team-style vector was fairly thin. The context-aware model had only a shallow relational layer, which meant it could not fully express pairwise football logic such as whether a winger fits a possession-heavy side, whether a club actually needs that role, or whether the player's age/value profile matches the target squad window. In the executed draft results, the context-aware model did outperform some baselines on ranking quality, but the gains were small and the contextual delta could even be negative depending on the exact rerun. In other words, the original pipeline was valid, but its **content/context layer was underpowered**.

That diagnosis motivated the work in this augmentation pass. The assignment brief requires all four recommender types, a clean comparison table, a simulated A/B mindset, and clear discussion of cold-start, bias, architecture, and business value. The master plan also emphasises tactical identity, squad gaps, and the need for richer player and team vectors. The new work therefore does not replace the baseline. Instead, it adds a **transparent synthetic feature layer** whose purpose is to enrich the proof-of-concept, increase explanatory power, and simulate the kind of richer internal data environment that real clubs often have but public student projects do not.

The synthetic layer was designed around a strict transparency rule. The project now preserves two benchmark families:

1. **Real-only benchmark**: the honest baseline built only from real observed data plus clearly engineered real-only features.
2. **Synthetic-augmented proof-of-concept benchmark**: a separate scenario-modelling track built from richer synthetic player, team, and team-player relational features.

Within the synthetic-augmented family, one additional distinction is important. There is first a **synthetic base** configuration, where synthetic features are added without label-informed uplift. There is then a **label-informed synthetic augmentation for proof-of-concept / scenario modelling** track, where selected pair-fit dimensions are moderately and probabilistically improved for actual observed transfer pairs. This distinction is essential. The synthetic PoC results are useful for storytelling, scouting simulation, and explaining how richer football context can improve ranking, but they are not presented as an unbiased estimate of out-of-sample generalisation.

In short, the inspection phase showed that the project already had a solid recommender skeleton and honest evaluation logic, but lacked a strong feature layer for football fit. The work in this report addresses that gap directly.

## 2. Data Preprocessing & Feature Engineering

The original preprocessing logic was retained wherever it was sound. Outfield players are filtered at 600 minutes, goalkeepers at 900 minutes, and real observed statistics are normalised within appropriate comparison groups. The existing draft already derived per-90 statistics, performed grouped imputation for selected fields, and used role clustering to assign more meaningful archetypes. Those steps remain defensible and are preserved as the real-only benchmark path.

The main contribution of this augmentation pass is a new **three-layer synthetic feature architecture**.

### 2.1 Player synthetic feature layer

The player synthetic layer does not generate independent random columns. Instead, it uses a **latent-factor design** anchored in the real data already present in the project. Each eligible player row receives a set of role-aware latent traits such as:

- finishing
- shot quality
- creativity
- progression
- ball carrying
- defensive intensity
- aerial / physicality
- press resistance
- reliability
- off-ball threat
- upside
- availability

These latents are estimated from a combination of real anchors, including goals, assists, shooting profile, tackles, interceptions, disciplinary profile, team success proxies, minutes share, market value, market-value change, age, and source-team context. They are also conditioned on interpretable role templates. For example, centre-forwards are given stronger priors on finishing, off-ball threat, aerial threat, and box occupation. Wide creators are given stronger priors on carrying, creativity, crossing, and progressive receiving. Controllers and advanced creators receive higher progression and press-resistance priors. Front-foot defenders and centre-backs receive stronger defensive and aerial priors.

The synthetic player observables are then derived from those latents. The saved augmented player file includes variables such as synthetic xG per 90, npxG per 90, xA per 90, xGI per 90, shot-creating actions, key passes, progressive passes, progressive carries, passes into the final third, passes into the box, successful crosses, tackles won, interceptions, aerial duels won, recoveries, pressures, pass completion, ball retention, availability percentage, plus-minus, goalkeeper shot-stopping, goalkeeper distribution, and several preference / contribution axes for explanation. Values are clipped to plausible football ranges and vary by role and minutes context. This prevents absurd combinations such as a classic centre-back producing winger-like dribbling and crossing outputs or a pure striker having elite midfield progression volume.

### 2.2 Team synthetic feature layer

The team synthetic layer is built from observed team information already available in the consolidated team file. It uses possession, shot profile, crosses, tackles, interceptions, fouls, cards, goal difference, points, goalkeeper concession rate, and squad market value to estimate a tactical latent space. The latent team dimensions include:

- possession orientation
- directness
- pressing intensity
- width / crossing tendency
- tempo
- territorial dominance
- defensive-line aggression
- transition dependence
- central-combination tendency
- development orientation

From these latents, the system derives more interpretable synthetic team observables such as possession percentage, passes per possession, build-up speed, average pass length, directness index, tempo index, width/crossing tendency, central-combination tendency, attacking-territory share, final-third entries, box entries, transition attacks, synthetic team xG, synthetic team xGA, pressing index, PPDA proxy, recovery height, defensive-line height, compactness, block depth, and budget-band index.

To make the style layer interpretable for demos and reporting, the team synthetic space is also clustered into four tactical archetypes:

- **aggressive_vertical_press**
- **patient_possession_control**
- **reactive_midblock_transition**
- **dominant_wide_control**

These labels are not decorative. They are used to explain why some players fit some clubs. A winger with strong carrying, wide delivery, and directness tolerance will score differently for an aggressive vertical press side than for a patient possession-control side.

### 2.3 Team-player relational / fit layer

The pair layer is the most important addition. The original context-aware recommender mostly combined collaborative and content scores with a few penalties. The augmented system introduces explicit pairwise football-fit variables:

- family need fit
- role fit
- style fit
- age fit
- budget fit
- availability fit
- readiness fit
- adaptation fit
- transfer success prior
- upside fit
- value-for-money fit
- tactical causality
- aggregate pair score

These variables are created for each team-candidate pair in the historical candidate set. The fit layer explicitly asks questions that actual scouts ask: does this club need this role, does the style match, is the player affordable relative to the club's budget band, is the age profile aligned with squad strategy, is there enough readiness for immediate minutes, and is there evidence that similar source-league or source-role transfers have worked before?

### 2.4 Label-informed synthetic augmentation

The proof-of-concept track goes one step further. For actual observed positive transfer pairs, selected pair-fit features are allowed to receive **moderate stochastic uplift**. This is deliberately disclosed and audited. The uplift affects style fit, need fit, budget fit, age fit, adaptation fit, upside fit, tactical causality, and transfer-success prior. It is not deterministic; there is noise and overlap with negatives, and the uplift is applied probabilistically rather than universally. In the test-season pair audit, the uplift flag is activated for about **83% of actual positive pairs** and **0% of non-actual pairs**. The mean positive-minus-negative gap in the aggregate pair score rises from **0.0393** in the baseline synthetic pair layer to **0.1008** in the PoC layer, which is material but not so extreme that the task becomes trivial.

This separation matters because it lets the project answer two different questions. The synthetic base track asks: *What if the recommender had richer football descriptors?* The synthetic PoC track asks: *What if we also explicitly model the fact that real successful transfers tend to exhibit better tactical, need, and affordability alignment than generic negatives?*

### 2.5 Saved outputs

The new augmentation workflow saves the following outputs:

- `players_master_synthetic_augmented.csv.gz`
- `teams_master_synthetic_augmented.csv.gz`
- `team_player_pair_synthetic_features.csv.gz`
- `synthetic_feature_dictionary.csv`
- `synthetic_generation_audit.json`
- metrics tables for real-only, synthetic-base, synthetic-PoC, and deltas
- demo shortlist files for named clubs
- sanity summary tables and figures

The workflow never overwrites the original raw project files. It writes into a new `data/synthetic/` directory and creates a separate synthetic notebook plus a separate augmented notebook.

## 3. Non-Personalized Recommender

The original non-personalised recommender is a role-aware composite ranking built from real observed features. It functions as a cold-start baseline and a sanity check. In the original draft, it performed poorly at low K because the same ranking is effectively shown to every club, which is especially weak in football because squad need and tactical identity matter enormously.

That pattern remains true in the reproduced real-only benchmark. The real-only non-personalised model records **Precision@10 = 0.0000**, **Recall@10 = 0.0000**, and **NDCG = 0.0000**. Its value is not ranking strength but baseline structure and cold-start fallback logic.

The augmentation pass introduces a synthetic non-personalised variant built from a **global transferability prior**. That score combines synthetic xGI, progression, upside, availability, mobility history, and value sensitivity to create a stronger global ranking of potentially interesting transfer targets. This does improve low-K performance somewhat: in the synthetic track the non-personalised variant reaches **Precision@10 = 0.0022**, **Recall@10 = 0.0063**, and **NDCG = 0.0030**. Those numbers are still modest, but that is the expected result. A global ranking should not beat a team-specific recommender in this domain. What matters is that the synthetic prior is now useful as a better universal cold-start surface and as a stronger seed list for later re-ranking.

From a business perspective, the non-personalised synthetic prior is also useful because it surfaces a more plausible “global scout board” than the original simple composite. A recruitment team could use it as an initial market scan before moving to the team-conditioned models.

## 4. Collaborative Filtering Recommender

The collaborative filtering logic in the original project is based on implicit feedback from team-player minutes. That is a sensible football analogy: clubs reveal preference by signing and playing footballers. The draft used ALS on a sparse team-player matrix aggregated from prior seasons. In the reproduced environment here, the same idea was implemented with an ALS-like factorisation fallback to preserve a reproducible path.

In the reproduced real-only benchmark, collaborative filtering remains one of the stronger models on the honest ranking task. It produces **Precision@10 = 0.0054**, **Recall@10 = 0.0165**, and **NDCG = 0.0083**, with high diversity and strong catalogue coverage. That is intuitive. CF captures hidden team affinities and recruitment network effects reasonably well.

However, collaborative filtering has a known football limitation: it does not know whether a candidate is tactically appropriate, affordable, or needed now. It can over-recommend players from high-exposure environments or lean too much on historical similarity.

The synthetic augmentation does not transform CF as dramatically as it transforms the content and context layers. In the synthetic base track, **Precision@10 remains 0.0054** and **Recall@10 improves only slightly to 0.0186**, while **NDCG slips slightly to 0.0076**. This is actually an important result. It suggests that richer player/team vectors mainly help the models that directly consume those vectors, whereas pure interaction-based factorisation is less sensitive to the new synthetic descriptors. The collaborative layer still benefits indirectly when the intrinsic player prior is added, but it does not become the headline winner.

This is useful analytically because it shows the augmentation is not just inflating every model indiscriminately. The strongest gains are concentrated where they should be: content-based similarity and context-aware re-ranking.

## 5. Content-Based Recommender

The content-based recommender is where the augmentation begins to pay off in a meaningful way. The original content model built team taste profiles from real player features and then used cosine similarity within roles. That framework is correct in principle, but it was constrained by the limited richness of the consolidated public feature set.

In the reproduced real-only benchmark, the content model reaches only **Precision@10 = 0.0011**, **Recall@10 = 0.0036**, and **NDCG = 0.0019**. This confirms that the original player vectors were too thin to express the kind of role/style nuance that transfer scouting needs.

The synthetic base track changes that substantially. Once player rows carry richer latent traits, synthetic observables, and linked team-style context, the content-based recommender improves to **Precision@10 = 0.0054**, **Recall@10 = 0.0251**, and **NDCG = 0.0109**. That is the clearest non-label-informed gain in the entire augmentation pass. NDCG improves by about **+0.0091** over the reproduced real-only content model, and recall rises sharply. This is exactly the pattern one would expect if the content vector had previously been underspecified.

The football explanation for the improvement is straightforward. A real-only public-data vector struggles to distinguish between, say, a wide creator forward and an all-action forward if the raw public columns are sparse or noisy. The synthetic layer forces the representation to encode structured football logic: ball carrying, creativity, progression, aeriality, pressing, off-ball threat, and contextual style preferences all become visible dimensions. The content engine can then retrieve players who are not merely similar in generic statistics, but similar in *how* they are likely to function in a team's tactical setup.

This has major demo value. It becomes possible to explain that a player is recommended because he offers high progression with strong press resistance for a possession-control side, or because he combines wide carry threat with crossing volume for a dominant wide-control side. That is far more persuasive than a generic cosine similarity score.

## 6. Context-Aware Recommender

The context-aware recommender is the final hybrid layer, and it is also where the synthetic augmentation has its largest proof-of-concept impact.

In the reproduced real-only pipeline, the context-aware model combines collaborative filtering and content similarity with a few context features. Its real-only results are respectable but not dominant: **Precision@10 = 0.0043**, **Recall@10 = 0.0129**, and **NDCG = 0.0072**. More importantly, its contextual delta relative to a nulled-context comparator is **negative (-0.0026)** in the reproduced setup. That is a clear sign that the original context layer was not yet strong enough. It was conceptually correct, but practically too shallow.

The synthetic augmentation fixes that weakness in two stages.

First, the **synthetic base** context-aware model replaces the thin context layer with the richer pair-fit architecture described earlier. Even without any label-informed uplift, the context-aware model improves to **NDCG = 0.0123** and achieves a **positive context delta of 0.0085**. That matters more than the raw headline number. It means the explicit pairwise football logic is now adding value rather than noise. The context model begins to behave like a proper scouting re-ranker rather than a loose weighted average of weak signals.

Second, the **synthetic PoC** context-aware model applies the labelled proof-of-concept uplift. In that track, the model reaches **Precision@10 = 0.0161**, **Recall@10 = 0.0485**, **NDCG = 0.0363**, and a **context delta of 0.0325**. Those are by far the best ranking metrics in the project, but they must be interpreted carefully. They demonstrate how strong the recommender can become in a richer, partially label-informed scenario space. They do not replace the honest baseline.

The decomposition matters. The synthetic base improvement shows that richer descriptors alone help. The additional jump from context-aware synthetic base (**NDCG 0.0123**) to context-aware synthetic PoC (**NDCG 0.0363**) shows that most of the big PoC gain is due to the explicitly uplifted pair layer. The audit confirms that the largest positive-vs-negative gap expansions occur in **need fit**, **tactical causality**, **budget fit**, **age fit**, **upside fit**, and especially the final **pair score**. This is exactly why the report must disclose the uplift and keep the tracks separated.

From a demo standpoint, however, the new context-aware layer is much stronger than the original notebook. It now supports explanations such as:

- the player matches the club's tactical style along possession/directness/pressing axes
- the player fills a specific role deficit in the squad
- the age profile is aligned with the club's current squad-building window
- the affordability/value picture is more plausible for the target budget band
- adaptation and readiness risks are visible rather than hidden

That explanatory richness is a major business and presentation improvement.

## 7. Comparative Evaluation

The evaluation remains time-based and historically structured. Training uses 2018-19 through 2021-22, validation uses 2022-23, and test uses 2023-24. The ground truth remains actual transfer arrivals weighted by target-season minutes. That logic is retained from the original project because it is appropriate and avoids obvious leakage into the honest benchmark.

For transparency, this project now keeps three evaluation tables.

### 7.1 Reproduced real-only benchmark

| Approach                                 |   RMSE |    MAE |   Precision@K |   Recall@K |   NDCG |   Coverage |   Diversity |   Serendipity | Context   |
|:-----------------------------------------|-------:|-------:|--------------:|-----------:|-------:|-----------:|------------:|--------------:|:----------|
| Non-Personalized                         | 0.4508 | 0.447  |        0      |     0      | 0      |     0.0065 |      0.6841 |        0      | —         |
| Collaborative Filtering (ALS-like rerun) | 0.4073 | 0.386  |        0.0054 |     0.0165 | 0.0083 |     0.3051 |      0.8142 |        0.003  | —         |
| Content-Based                            | 0.5337 | 0.5    |        0.0011 |     0.0036 | 0.0019 |     0.2501 |      0.6227 |        0      | —         |
| Context-Aware                            | 0.4634 | 0.4376 |        0.0043 |     0.0129 | 0.0072 |     0.2965 |      0.7334 |        0.0027 | -0.0026   |

The honest real-only picture is clear. Collaborative filtering is the strongest model on NDCG in the reproduced baseline, while context-aware is competitive but not decisively better. The content-based model remains weak because the real-only feature vectors are too limited. That honest weakness is important because it motivates the augmentation rather than being hidden by it.

### 7.2 Synthetic base benchmark (richer vectors only)

| Approach                                |   RMSE |    MAE |   Precision@K |   Recall@K |   NDCG |   Coverage |   Diversity |   Serendipity | Context   |
|:----------------------------------------|-------:|-------:|--------------:|-----------:|-------:|-----------:|------------:|--------------:|:----------|
| Non-Personalized (synthetic)            | 0.4747 | 0.445  |        0.0022 |     0.0063 | 0.003  |     0.0065 |      0.5566 |        0.0012 | —         |
| Collaborative Filtering + synthetic fit | 0.4903 | 0.465  |        0.0054 |     0.0186 | 0.0076 |     0.1294 |      0.7092 |        0.0017 | —         |
| Content-Based + synthetic fit           | 0.5358 | 0.5034 |        0.0054 |     0.0251 | 0.0109 |     0.1876 |      0.6432 |        0.0011 | —         |
| Context-Aware (synthetic base)          | 0.5333 | 0.507  |        0.0022 |     0.0134 | 0.0123 |     0.1121 |      0.7046 |        0.0002 | 0.0085    |

This table demonstrates what richer football representations do *before* any label-informed uplift. The standout gain is the content-based model, whose NDCG increases from 0.0019 to 0.0109. The context-aware model also improves from 0.0072 to 0.0123 and, crucially, flips from a negative to a positive context delta. Collaborative filtering changes much less, which again supports the interpretation that the richer vectors are mainly helping the content and context layers.

### 7.3 Synthetic proof-of-concept benchmark

| Approach                                |   RMSE |    MAE |   Precision@K |   Recall@K |   NDCG |   Coverage |   Diversity |   Serendipity | Context   |
|:----------------------------------------|-------:|-------:|--------------:|-----------:|-------:|-----------:|------------:|--------------:|:----------|
| Non-Personalized (synthetic)            | 0.4747 | 0.445  |        0.0022 |     0.0063 | 0.003  |     0.0065 |      0.5566 |        0.0012 | —         |
| Collaborative Filtering + synthetic fit | 0.4903 | 0.465  |        0.0054 |     0.0186 | 0.0076 |     0.1294 |      0.7092 |        0.0017 | —         |
| Content-Based + synthetic fit           | 0.5358 | 0.5034 |        0.0054 |     0.0251 | 0.0109 |     0.1876 |      0.6432 |        0.0011 | —         |
| Context-Aware (synthetic PoC)           | 0.5355 | 0.5092 |        0.0161 |     0.0485 | 0.0363 |     0.1116 |      0.7056 |        0.0049 | 0.0325    |

The context-aware synthetic PoC model is the highest-performing system in the project by a wide margin. Precision, recall, NDCG, serendipity, and contextual lift all improve strongly. This is a useful proof-of-concept because it shows the upside of richer, more causally informed scouting features. But it should be read as **scenario modelling**, not as the core honest benchmark.

### 7.4 Delta view

| Track                       | Approach                                |   Δ Precision@K |   Δ Recall@K |   Δ NDCG |   Δ Coverage |   Δ Diversity |   Δ Serendipity | Δ Context   |
|:----------------------------|:----------------------------------------|----------------:|-------------:|---------:|-------------:|--------------:|----------------:|:------------|
| Synthetic base vs real-only | Non-Personalized (synthetic)            |          0.0022 |       0.0063 |   0.003  |       0      |       -0.1274 |          0.0012 | —           |
| Synthetic base vs real-only | Collaborative Filtering + synthetic fit |          0      |       0.0021 |  -0.0006 |      -0.1757 |       -0.105  |         -0.0013 | —           |
| Synthetic base vs real-only | Content-Based + synthetic fit           |          0.0043 |       0.0215 |   0.0091 |      -0.0625 |        0.0205 |          0.0011 | —           |
| Synthetic base vs real-only | Context-Aware (synthetic base)          |         -0.0022 |       0.0005 |   0.0051 |      -0.1844 |       -0.0288 |         -0.0025 | 0.0111      |
| Synthetic PoC vs real-only  | Non-Personalized (synthetic)            |          0.0022 |       0.0063 |   0.003  |       0      |       -0.1274 |          0.0012 | —           |
| Synthetic PoC vs real-only  | Collaborative Filtering + synthetic fit |          0      |       0.0021 |  -0.0006 |      -0.1757 |       -0.105  |         -0.0013 | —           |
| Synthetic PoC vs real-only  | Content-Based + synthetic fit           |          0.0043 |       0.0215 |   0.0091 |      -0.0625 |        0.0205 |          0.0011 | —           |
| Synthetic PoC vs real-only  | Context-Aware (synthetic PoC)           |          0.0118 |       0.0356 |   0.0291 |      -0.1849 |       -0.0278 |          0.0022 | 0.0351      |

The delta view shows the underlying story more cleanly than the raw tables. The biggest non-label-informed uplift is content-based NDCG. The biggest overall uplift is context-aware NDCG in the synthetic PoC track. Collaborative filtering barely improves, which is analytically healthy. It means the synthetic layer is not simply inflating every number; it is helping the layers that actually consume football context.

### 7.5 Why the metrics improved

The metric changes come from two different mechanisms.

**Mechanism A — richer vectors.** The synthetic base track adds informative player and team descriptors that the real-only pipeline did not have. This mainly benefits content-based similarity and context-aware pair scoring. It also improves explanations because recommendations can now be tied to explicit football dimensions rather than generic composite scores.

**Mechanism B — label-informed pair engineering.** The synthetic PoC track introduces moderate uplift on selected pair-fit dimensions for actual positive pairs. This is the main reason the context-aware model jumps from NDCG 0.0123 in synthetic base to NDCG 0.0363 in synthetic PoC. The audit file makes this explicit rather than hiding it. The pair-score gap between actual and non-actual pairs increases from **0.0393** to **0.1008** after PoC uplift. The strongest uplifted dimensions are **need fit**, **tactical causality**, and **budget/age/upside-related fit variables**.

### 7.6 Named-club demos

The augmented notebook also includes named-club demo files. For Arsenal, the PoC shortlist surfaces **Declan Rice** within the top recommendations, which is useful because it aligns with a known real arrival while still providing explanation variables. For Girona, the shortlist is affordability-filtered to produce a more plausible smaller-budget recruitment story. These demos are not formal evaluation, but they substantially improve the presentation value of the system.

Overall, the evaluation now tells a much richer and more honest story than the original draft. The real-only benchmark remains the core reference point. The synthetic base track shows what better football descriptors can do. The synthetic PoC track shows the upper-bound storytelling potential of a richer, partly label-informed scouting environment.

## 8. Business Case & Deployment Design

The business logic of the project is stronger after augmentation because the model output is no longer just a ranking; it is a ranking with football rationale.

In a real club environment, the recommender would not replace scouts. It would widen the search universe, surface candidate clusters that human analysts may miss, and provide a structured rationale for why a player belongs on a shortlist. That is particularly valuable for clubs that cannot maintain large international scouting networks. A mid-tier club cannot watch every second-tier winger in Europe, but it can use a recommender to narrow the field to a manageable board of plausible fits.

The synthetic layer strengthens that story because it approximates the type of richer feature environment that proprietary systems often have. Public student datasets rarely include reliable tracking-based off-ball movement, true pressure data, or full event streams across every target league. Real clubs do often have richer context than that. The synthetic layer therefore acts as a **scenario model for a more mature scouting stack**, not as a claim that public data magically became complete.

Commercially, the strongest argument remains transfer-risk reduction and bargain identification. A recommender that improves shortlist quality can reduce the probability of expensive recruitment misses and increase the probability of identifying undervalued players before the broader market reacts. Even one avoided failed signing or one correctly identified value opportunity can justify the cost of a small analytics function. The augmented context-aware model is especially useful here because it can articulate *why* a player is a fit: tactical alignment, squad need, age window, affordability, and adaptation profile.

From a deployment perspective, the system architecture is still consistent with the master plan. A production version would include raw data ingestion, a feature store, scheduled retraining, serialised recommender artefacts, a serving API, monitoring, and a scout-facing interface. The new synthetic layer fits naturally into that architecture as an additional feature-generation stage with audit logging. In a production setting, the synthetic layer would likely be replaced or supplemented by richer proprietary data feeds rather than kept as a permanent source of label-informed uplift. But as a student proof-of-concept, it demonstrates clearly what that richer layer would do for the final ranking and for explanation quality.

## 9. Cold-Start & Bias Mitigation Strategy

Cold-start remains structural in football. Young players with sparse senior minutes are often the most interesting transfer targets and the hardest to model. The synthetic layer helps modestly here because it includes upside, age-curve, and development-orientation priors. Those do not solve cold-start, but they make the system less brittle than a pure real-stat representation.

Newly promoted clubs and clubs with thin historical interaction data are another challenge. The non-personalised synthetic prior becomes a stronger fallback than the original simple composite. The team synthetic layer also makes it easier to initialise a new club with a style-archetype hypothesis rather than no style information at all.

Bias remains an important limitation. The real interaction matrix is still dominated by top-league behaviour. Public market values are imperfect and lagged. Actual transfers reflect managerial preferences, agents, contract situations, and work-permit constraints that are not fully visible in the data. The synthetic layer introduces an additional bias risk: if it were not disclosed, readers might mistake uplifted PoC metrics for fair held-out performance. This report explicitly avoids that mistake.

The mitigation strategy is therefore partly technical and partly communicative. Technically, the project keeps separate tracks, preserves the original benchmark, logs uplift dimensions, and stores an audit file. Communicatively, the report repeatedly states that the label-informed synthetic track is a **proof-of-concept / scenario model**. That matters because transparency is part of the methodological contribution, not an appendix detail.

A second mitigation is that uplift was deliberately kept moderate rather than making the task trivial. The actual-vs-negative distributions still overlap. Not every real signing is forced to the top. That preserves enough ambiguity that the PoC remains illustrative rather than purely tautological.

## 10. Conclusions

The project inspection confirmed that the original football transfer recommender already had the right overall structure but lacked a sufficiently rich football context layer. The new augmentation pass addresses that gap with a transparent synthetic architecture built across player, team, and team-player relational levels.

The most important conclusion is methodological. The **real-only benchmark remains the core honest evaluation** and should stay that way in the final submission. In that benchmark, collaborative filtering remains strong, content-based is weak, and context-aware is only moderately successful. That is the fair baseline.

The second conclusion is that **richer football descriptors do help**, even before any label-informed uplift. The synthetic base track substantially improves the content-based recommender and makes the context-aware model genuinely contextual, as shown by its positive context delta. This is a meaningful proof that the original weakness was largely feature-related, not just model-related.

The third conclusion is that **label-informed pair engineering can produce a much more impressive proof-of-concept**. The synthetic PoC context-aware model is clearly the strongest system in ranking terms, explanation quality, and demo value. But that gain comes precisely because the pair layer is allowed to become more favourable for actual transfer pairs. That makes it useful for scenario modelling and presentation, but it must not replace the honest held-out benchmark.

From a football perspective, the augmented system is much closer to the type of shortlist generator that clubs actually want. It can tell a story about style fit, role fit, need fit, affordability, and tactical causality rather than just outputting a score. That is a significant improvement in practical relevance.

Future work should focus on three paths. First, expand the real feature set with richer event data or tracking-derived proxies if available. Second, replace the heuristic hybrid with a more rigorous context model such as a factorisation machine or graph-based approach. Third, keep the current synthetic layer as a prototyping tool, but progressively substitute its functions with richer observed data sources. That would let the project retain the explanatory power demonstrated here while moving closer to a fully defensible real-world benchmark.

## 11. Individual Contributions

The table below is a placeholder aligned with the course requirement and should be updated with the final team member names before submission.

| Member | Primary responsibilities | Secondary contributions |
|---|---|---|
| Member 1 | Data acquisition, EDA, original domain framing | Report §1, visual QA |
| Member 2 | Collaborative filtering pipeline and sparse interaction logic | Report §4, evaluation plumbing |
| Member 3 | Content-based recommender, role clustering, feature engineering | Report §2, §5 |
| Member 4 | Context-aware recommender, synthetic pair layer, tactical interpretation | Report §6, deployment design |
| Member 5 | Evaluation framework, benchmark comparison, A/B simulation, presentation integration | Report §7, §8, §9, §10 |

For this augmentation pass specifically, the additional workload should be allocated across synthetic player generation, synthetic team generation, pair-fit engineering, audit and disclosure writing, notebook augmentation, and report integration.

## 12. AI Usage Disclosure

**Tools used.** ChatGPT was used to inspect the uploaded project materials, refactor the notebook workflow, generate transparent synthetic feature engineering code, draft report text, and package the new artifacts.

**Prompts and tasks.** AI assistance was used for: reverse engineering the existing notebook structure; proposing and implementing a controlled synthetic player/team/pair feature layer; producing benchmark comparison tables; generating audit and disclosure text; drafting the executive summary and report; and packaging the new synthetic notebook plus augmented notebook.

**Verification.** All numerical results were computed from notebook/script code, not invented in prose. Synthetic uplift dimensions were logged explicitly in the audit file. The real-only and synthetic-augmented tracks were kept separate in both code and reporting. The team remains responsible for reviewing all code, checking file outputs, validating the metrics tables, and editing the prose for the final submission.

**Responsibility.** All team members remain fully responsible for any errors in the submission.

## Synthetic data disclosure paragraph

This project includes a clearly flagged **synthetic augmentation layer** used to enrich player, team, and team-player fit features for proof-of-concept purposes. The synthetic-augmented results are presented separately from the honest real-only benchmark. Within the synthetic track, one scenario explicitly applies **label-informed uplift** to selected pair-fit dimensions for observed positive transfer pairs. These uplifted results are disclosed as **proof-of-concept / scenario modelling** and are not claimed to represent unbiased real-world generalisation.