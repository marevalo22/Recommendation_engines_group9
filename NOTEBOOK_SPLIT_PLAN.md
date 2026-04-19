# Optional Plan for Splitting the Current Draft into the Final 5-Notebook Course Structure

## Proposed final structure

### 01_non_personalized.ipynb
- domain recap
- EDA essentials
- real-only preprocessing
- non-personalized ranking
- cold-start fallback logic

### 02_collaborative_filtering.ipynb
- interaction matrix
- ALS / implicit feedback logic
- validation tuning
- team-level recommendation demos

### 03_content_based.ipynb
- role clustering
- real-only player vectors
- synthetic player vector appendix or import
- content-based retrieval
- role-based similarity demos

### 04_context_aware.ipynb
- team style vectors
- synthetic team tactical layer
- pairwise fit features
- hybrid scoring logic
- explanation outputs

### 05_evaluation.ipynb
- preserved original benchmark
- reproduced real-only benchmark
- synthetic-base table
- synthetic-PoC table
- standardised comparison table
- A/B simulation discussion
- beyond-accuracy metrics

## Recommended packaging rule

Keep the synthetic-generation notebook (`00_synthetic_feature_layer.ipynb`) as a supporting methodology notebook outside the final 01–05 sequence, or fold its generation code into a small reusable helper module imported by 03, 04, and 05.

## Suggested section ownership

- 01 + data EDA: domain/data owners
- 02: CF owner
- 03: content / feature engineering owner
- 04: context-aware / synthetic-fit owner
- 05: evaluation + report integration owner