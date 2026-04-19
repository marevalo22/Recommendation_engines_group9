# Restructuring Note

## New folder structure

The project now contains a `final_submission/` folder with only the files intended for direct submission:
- 01_non_personalized.ipynb / .py
- 02_collaborative_filtering.ipynb / .py
- 03_content_based.ipynb / .py
- 04_context_aware.ipynb / .py
- 05_evaluation.ipynb / .py
- GROUP_9_report.md
- GROUP_9_executive_summary.md

All other project materials remain outside that folder, including:
- real data
- synthetic data
- helper modules
- support notebooks
- changelog and disclosure notes
- original draft report files

## Old to new mapping

| Existing file | Submission-ready destination | Notes |
|---|---|---|
| football_recsys_draft_augmented.ipynb | final_submission/05_evaluation.ipynb and support overview notebook | Evaluation content split and cleaned |
| 00_synthetic_feature_layer.ipynb | kept as support notebook outside final_submission | Not one of the 5 mandatory notebooks |
| synthetic_recsys_helpers.py | kept as support helper outside final_submission | Original augmentation helper retained |
| GROUP_report_draft_synthetic_augmented.md | final_submission/GROUP_9_report.md | Mandatory headings preserved; section 11 updated |
| GROUP_executive_summary_draft.md | final_submission/GROUP_9_executive_summary.md | Renamed to rubric format |
| metrics_* CSVs in data/synthetic | consumed by final_submission/05_evaluation.ipynb | Honest real-only and synthetic PoC kept separate |

## Notebook content split

| Final notebook | Main content moved into it |
|---|---|
| 01_non_personalized.ipynb | domain framing, dataset overview, preprocessing summary, role-aware non-personalized baseline |
| 02_collaborative_filtering.ipynb | interaction matrix, latent-factor collaborative filtering, club examples |
| 03_content_based.ipynb | feature engineering, role/style representation, player similarity, team content-based recommendations |
| 04_context_aware.ipynb | team context, synthetic team-player fit features, tactical archetypes, shortlist explanations |
| 05_evaluation.ipynb | preserved original benchmark, honest real-only benchmark, synthetic base/PoC comparisons, delta analysis, final standardized table |
