# Executive Summary

This project develops an end-to-end recommendation system for the football transfer market, where clubs are treated as users and players as items, and transfers with minutes played serve as implicit feedback. The objective is to generate shortlists of players that a club should consider signing during a transfer window.

The problem is inherently challenging due to extreme sparsity: each club signs only a few players per season from a large candidate pool. This results in a sparse interaction matrix (~1% density) and very limited positive labels, making accurate ranking difficult. Additionally, publicly available data provides a relatively narrow feature space, constraining the ability of feature-based models to capture tactical and contextual nuances.

Four recommender approaches were implemented and evaluated:

- A non-personalized model, which produces global position-specific rankings based on weighted performance statistics  
- A collaborative filtering model, using TruncatedSVD on a log-transformed interaction matrix  
- A content-based model, representing players as feature vectors and matching them to team profiles  
- A context-aware model, incorporating synthetic features to capture tactical style, squad needs, and budget constraints  

Evaluation was conducted using a strict temporal split, with the 2023–24 season held out as the test set. Metrics included Precision@10, Recall@10, NDCG, RMSE, coverage, diversity, and serendipity.

The results show that collaborative filtering performs best, achieving the highest ranking accuracy (NDCG ≈ 0.0083, Precision@10 ≈ 0.0054). The non-personalized model performs poorly on ranking metrics, as it ignores club-specific needs, but remains useful as a baseline and cold-start solution. The content-based model is limited by the narrow feature space, while the context-aware model does not outperform simpler approaches in the real-only benchmark due to reliance on synthetic features.

A key finding is that feature richness is the main bottleneck. When richer synthetic descriptors are introduced, performance improves for feature-based models, indicating that limitations are primarily data-driven rather than methodological. However, synthetic enhancements are treated as exploratory and are not used as the primary evaluation benchmark.

From a business perspective, the system provides value as a decision-support tool, reducing a large candidate pool to a short, interpretable list of potential signings. This is particularly beneficial for mid-tier clubs with limited scouting resources, enabling more efficient and data-driven recruitment processes.

Future work should focus on incorporating richer real-world data, improving contextual modeling, and replacing synthetic proxies with observed features to achieve stronger and more reliable performance.
