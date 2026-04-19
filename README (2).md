# Football Transfer Market Recommender System

## Overview

This project develops a recommender system for the football transfer market, where **clubs are treated as users**, **players are treated as items**, and **transfers with minutes played are used as implicit feedback**. The aim is to generate data-driven shortlists of players that a club could consider signing during a transfer window. fileciteturn0file0L8-L15

The recommendation task is intentionally difficult. The dataset is highly sparse because each club signs only a small number of players per season relative to a large candidate pool, which makes accurate ranking challenging. The project therefore evaluates several recommender approaches rather than relying on a single method. fileciteturn0file0L12-L15

## Project Objective

The main objective of the project is to test whether recommender-system methods can be adapted to support football recruitment by identifying players who are plausible transfer targets for specific clubs. The system is designed as a **decision-support tool**, not a replacement for scouts or sporting directors. It is intended to reduce a large candidate pool into a more interpretable shortlist. fileciteturn0file1L3-L12

## Data Sources

The project combines football data from two main sources:

- **FBref** via the `soccerdata` library for player- and team-level performance statistics across six seasons
- **Transfermarkt** for player market values and transfer history proxies fileciteturn0file0L17-L24

The final working dataset reported in the project includes:

- **16,873 player-season rows**
- **586 team-season rows**
- **6,021 unique players**
- **140 unique clubs** fileciteturn0file0L28-L39

After minimum-minutes filtering, the player-season dataset is reduced to **11,093 rows** to improve feature reliability. fileciteturn0file0L63-L66

## Deliverables Included

This final deliverable contains the core written outputs and cleaned master datasets used in the project:

- `Report_REC.md` — full final report
- `executive_summary.md` — condensed summary of the project
- `players_master_with_market_values.csv` — consolidated player-level dataset with market values
- `teams_master_with_market_values.csv` — consolidated team-level dataset with market values
- `README.md` — project overview and usage guide

## Recommendation Approaches Implemented

The project evaluates four approaches:

### 1. Non-Personalized Recommender
A global, position-specific ranking model based on weighted football performance statistics. It acts as a baseline and cold-start fallback. fileciteturn0file0L102-L109

### 2. Collaborative Filtering
A latent-factor recommender using **TruncatedSVD** on a sparse club–player interaction matrix built from log-transformed minutes played. This is the strongest model in the real-only benchmark. fileciteturn0file0L159-L177

### 3. Content-Based Recommender
A feature-vector model where players are represented through performance statistics, role encoding, and age terms, and clubs are represented through squad-profile centroids. Recommendations are produced through cosine similarity. fileciteturn0file0L213-L234

### 4. Context-Aware Recommender
A scoring model that adds tactical style, squad needs, and budget compatibility through synthetic contextual features. This was treated as an exploratory proof of concept rather than the main benchmark. fileciteturn0file0L251-L278
## Evaluation Design

The project uses a **strict temporal split** to avoid leakage:

- **Training:** 2018–19 to 2021–22
- **Validation:** 2022–23
- **Test:** 2023–24 fileciteturn0file0L40-L44

Models are compared using:

- Precision@10
- Recall@10
- NDCG@10
- RMSE
- MAE
- Coverage
- Diversity
- Serendipity fileciteturn0file0L289-L299

## Main Results

The core result of the project is that **collaborative filtering performs best on the honest real-only benchmark**. The reported results are:

| Model | Precision@10 | Recall@10 | NDCG |
|---|---:|---:|---:|
| Non-Personalized | 0.0000 | 0.0000 | 0.0000 |
| Collaborative Filtering | 0.0054 | 0.0165 | 0.0083 |
| Content-Based | 0.0011 | 0.0036 | 0.0019 |
| Context-Aware | 0.0043 | 0.0129 | 0.0072 | fileciteturn0file0L301-L320

This means the project’s strongest conclusion is that **interaction-based methods currently outperform feature-based methods under the available real-world data constraints**. At the same time, the report concludes that **feature richness is the main bottleneck**, because richer synthetic descriptors improved feature-based models in exploratory settings. fileciteturn0file0L321-L336

## Key Business Interpretation

The system is most useful as a **shortlist-generation tool**. Its value lies in helping clubs reduce search space, support recruitment discussions, and identify players that could otherwise be missed. This is particularly relevant for clubs with smaller scouting departments and fewer resources. fileciteturn0file1L13-L16

## File Usage Guide

### Full Report
Open `Report_REC.md` for the complete methodology, results, business case, limitations, and conclusions.

### Executive Summary
Open `executive_summary.md` for a short version suitable for quick review or presentation context.

### Datasets
- Use `players_master_with_market_values.csv` for player-level modelling, feature engineering, and recommendation experiments.
- Use `teams_master_with_market_values.csv` for team-level context, club information, and any team-based enrichment.

## Methodological Notes

- Real and synthetic features were kept separate where possible to preserve transparency in evaluation. fileciteturn0file0L92-L100
- The real-only benchmark is treated as the primary result.
- Synthetic features are included as exploratory extensions and should not be interpreted as equally validated production signals. fileciteturn0file0L360-L373

## Limitations

Key limitations acknowledged in the project include:

- Extreme sparsity in the interaction matrix
- Limited publicly available feature richness
- Low absolute ranking metrics due to the difficulty of the task
- Use of synthetic contextual variables for exploratory modelling rather than fully observed real-world context fileciteturn0file0L12-L15 fileciteturn0file0L331-L336

## Authors and Contribution Split

The report attributes the work across the team as follows:

- Andrea Saxod — domain analysis, non-personalized model
- Cloe Chapotot — collaborative filtering, notebook QA
- Constantin Hatecke — content-based model, feature engineering
- Marcela Funabishi — context-aware model
- Matias Arevalo — evaluation and metrics
- Vittorio Fialdini — business case, integration, packaging fileciteturn0file0L388-L398

## Final Conclusion

This project shows that football transfer recommendation is a valid but difficult recommender-systems problem. The strongest real-only result comes from collaborative filtering, while richer features appear to be the main path for future improvement. The current system is therefore best positioned as an interpretable, data-assisted scouting support tool rather than a fully automated recruitment engine. fileciteturn0file0L375-L386
