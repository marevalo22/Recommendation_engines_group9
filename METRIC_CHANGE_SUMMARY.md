# Metric Change Summary

## Reproduced real-only benchmark

- Best real-only NDCG@10: **Collaborative Filtering = 0.0083**
- Real-only Context-Aware NDCG@10: **0.0072**
- Real-only Content-Based NDCG@10: **0.0019**

## Synthetic base (richer vectors only)

- Content-Based NDCG@10 rises from **0.0019** to **0.0109** (**+0.0091**)
- Context-Aware NDCG@10 rises from **0.0072** to **0.0123** (**+0.0051**)
- Context delta moves from **-0.0026** to **0.0085**

## Synthetic PoC (label-informed scenario model)

- Context-Aware Precision@10 improves from **0.0043** to **0.0161**
- Context-Aware Recall@10 improves from **0.0129** to **0.0485**
- Context-Aware NDCG@10 improves from **0.0072** to **0.0363** (**+0.0291**)
- Context delta rises to **0.0325**

## Interpretation

- The biggest **non-label-informed** gain comes from richer vectors helping the **content-based** model.
- The strongest **overall** gain comes from the **context-aware synthetic PoC** track, where the pair layer receives explicit label-informed uplift.
- Collaborative filtering changes only slightly, which suggests the synthetic layer primarily benefits models that directly use football descriptors rather than only interaction history.