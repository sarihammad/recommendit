"""
Recommendation evaluation metrics.
Implements NDCG@K, Recall@K, MRR, catalog coverage,
training-serving skew detection, and a full evaluation report.
"""
import logging
import math
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
# Core Ranking Metrics                                                 #
# ------------------------------------------------------------------ #

def ndcg_at_k(
    recommended: List[Any],
    relevant: List[Any],
    k: int,
    relevance_scores: Optional[Dict[Any, float]] = None,
) -> float:
    """
    Normalized Discounted Cumulative Gain at K.

    Args:
        recommended: ranked list of item IDs (best first)
        relevant: set of relevant item IDs (ground truth)
        k: cutoff position
        relevance_scores: optional dict mapping item_id → graded relevance.
            If None, binary relevance (1 if in relevant set, else 0) is used.

    Returns:
        NDCG@K score in [0, 1]
    """
    relevant_set = set(relevant)
    top_k = recommended[:k]

    def get_rel(item) -> float:
        if relevance_scores is not None:
            return float(relevance_scores.get(item, 0.0))
        return 1.0 if item in relevant_set else 0.0

    # DCG
    dcg = 0.0
    for i, item in enumerate(top_k):
        rel = get_rel(item)
        if rel > 0:
            dcg += rel / math.log2(i + 2)  # i+2 because log2(1) = 0

    # Ideal DCG: sort by relevance
    if relevance_scores is not None:
        ideal_rels = sorted(
            [relevance_scores.get(item, 0.0) for item in relevant],
            reverse=True,
        )[:k]
    else:
        ideal_rels = [1.0] * min(len(relevant), k)

    idcg = sum(
        rel / math.log2(i + 2) for i, rel in enumerate(ideal_rels) if rel > 0
    )

    if idcg == 0:
        return 0.0
    return dcg / idcg


def recall_at_k(
    recommended: List[Any],
    relevant: List[Any],
    k: int,
) -> float:
    """
    Recall at K: fraction of relevant items found in the top-K recommendations.

    Returns:
        Recall@K in [0, 1]
    """
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in relevant_set)
    return hits / len(relevant_set)


def precision_at_k(
    recommended: List[Any],
    relevant: List[Any],
    k: int,
) -> float:
    """Precision at K: fraction of top-K recommendations that are relevant."""
    if k == 0:
        return 0.0
    relevant_set = set(relevant)
    hits = sum(1 for item in recommended[:k] if item in relevant_set)
    return hits / k


def mrr(
    recommended: List[Any],
    relevant: List[Any],
) -> float:
    """
    Mean Reciprocal Rank.
    Returns the reciprocal rank of the first relevant item in the recommendation list.

    Returns:
        MRR score in [0, 1]
    """
    relevant_set = set(relevant)
    for rank, item in enumerate(recommended, start=1):
        if item in relevant_set:
            return 1.0 / rank
    return 0.0


def average_precision(
    recommended: List[Any],
    relevant: List[Any],
) -> float:
    """Average Precision: area under the Precision-Recall curve."""
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    hits = 0
    sum_precisions = 0.0
    for i, item in enumerate(recommended, start=1):
        if item in relevant_set:
            hits += 1
            sum_precisions += hits / i
    return sum_precisions / len(relevant_set)


# ------------------------------------------------------------------ #
# Catalog Coverage                                                     #
# ------------------------------------------------------------------ #

def coverage(
    all_recommendations: List[List[Any]],
    catalog_size: int,
) -> float:
    """
    Catalog coverage: fraction of the item catalog that appears
    in at least one recommendation list.

    Args:
        all_recommendations: list of recommendation lists (one per user)
        catalog_size: total number of items in the catalog

    Returns:
        Coverage ratio in [0, 1]
    """
    if catalog_size == 0:
        return 0.0
    recommended_items = set()
    for recs in all_recommendations:
        recommended_items.update(recs)
    return len(recommended_items) / catalog_size


def intra_list_diversity(
    recommendations: List[Any],
    item_genre_vectors: Dict[Any, np.ndarray],
) -> float:
    """
    Average pairwise distance between recommended items' genre vectors.
    Higher values indicate more diverse recommendations.
    """
    if len(recommendations) < 2:
        return 0.0
    vecs = [item_genre_vectors.get(iid) for iid in recommendations if iid in item_genre_vectors]
    if len(vecs) < 2:
        return 0.0
    total_dist = 0.0
    count = 0
    for i in range(len(vecs)):
        for j in range(i + 1, len(vecs)):
            # 1 - cosine_similarity
            v1, v2 = vecs[i], vecs[j]
            norm1 = np.linalg.norm(v1)
            norm2 = np.linalg.norm(v2)
            if norm1 > 0 and norm2 > 0:
                cos_sim = np.dot(v1, v2) / (norm1 * norm2)
                total_dist += 1 - cos_sim
                count += 1
    return total_dist / count if count > 0 else 0.0


# ------------------------------------------------------------------ #
# Training-Serving Skew Detection                                      #
# ------------------------------------------------------------------ #

def kl_divergence_bins(
    p_values: np.ndarray,
    q_values: np.ndarray,
    n_bins: int = 20,
    epsilon: float = 1e-10,
) -> float:
    """
    Estimate KL divergence KL(P || Q) using histogram binning.

    Args:
        p_values: samples from the training distribution
        q_values: samples from the serving distribution
        n_bins: number of histogram bins
        epsilon: smoothing constant to avoid log(0)

    Returns:
        KL divergence (>0; higher = more different distributions)
    """
    # Use combined range for consistent binning
    combined = np.concatenate([p_values, q_values])
    bin_min, bin_max = combined.min(), combined.max()
    if bin_min == bin_max:
        return 0.0

    edges = np.linspace(bin_min, bin_max, n_bins + 1)
    p_hist, _ = np.histogram(p_values, bins=edges, density=True)
    q_hist, _ = np.histogram(q_values, bins=edges, density=True)

    # Normalize to proper probability distributions
    p_hist = p_hist + epsilon
    q_hist = q_hist + epsilon
    p_hist = p_hist / p_hist.sum()
    q_hist = q_hist / q_hist.sum()

    # KL(P || Q) = sum(p * log(p/q))
    kl = np.sum(p_hist * np.log(p_hist / q_hist))
    return float(kl)


def detect_training_serving_skew(
    train_features_df: pd.DataFrame,
    serving_features_df: pd.DataFrame,
    threshold: float = 0.1,
    numeric_cols: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Detect training-serving skew by comparing feature distributions.
    Uses KL divergence per feature and flags features exceeding the threshold.

    Args:
        train_features_df: feature DataFrame from training time
        serving_features_df: feature DataFrame from serving (recent requests)
        threshold: KL divergence threshold above which a feature is flagged
        numeric_cols: list of columns to check; defaults to all numeric columns

    Returns:
        Dict with keys:
          - 'feature_kl': dict of {feature_name: kl_divergence}
          - 'flagged_features': list of features with kl > threshold
          - 'max_kl': maximum KL divergence observed
          - 'skew_detected': bool
    """
    if numeric_cols is None:
        numeric_cols = [
            c for c in train_features_df.select_dtypes(include=[np.number]).columns
            if c in serving_features_df.columns
        ]

    feature_kl: Dict[str, float] = {}
    for col in numeric_cols:
        train_vals = train_features_df[col].dropna().values.astype(float)
        serving_vals = serving_features_df[col].dropna().values.astype(float)
        if len(train_vals) < 10 or len(serving_vals) < 10:
            continue
        kl = kl_divergence_bins(train_vals, serving_vals)
        feature_kl[col] = round(kl, 6)

    flagged = [feat for feat, kl_val in feature_kl.items() if kl_val > threshold]
    max_kl = max(feature_kl.values()) if feature_kl else 0.0

    result = {
        "feature_kl": feature_kl,
        "flagged_features": flagged,
        "max_kl": max_kl,
        "skew_detected": len(flagged) > 0,
        "threshold": threshold,
        "n_features_checked": len(feature_kl),
    }

    if flagged:
        logger.warning(
            "Training-serving skew detected in %d features: %s",
            len(flagged), flagged[:5],
        )
    else:
        logger.info("No significant training-serving skew detected.")

    return result


# ------------------------------------------------------------------ #
# Full Evaluation Report                                               #
# ------------------------------------------------------------------ #

def evaluate_model(
    recommendations_by_user: Dict[Any, List[Any]],
    ground_truth_by_user: Dict[Any, List[Any]],
    k_values: List[int] = None,
    catalog_size: Optional[int] = None,
    item_genre_vectors: Optional[Dict[Any, np.ndarray]] = None,
) -> Dict[str, Any]:
    """
    Compute a full evaluation report across multiple K values.

    Args:
        recommendations_by_user: dict mapping user_id → ranked list of recommended item_ids
        ground_truth_by_user: dict mapping user_id → list of relevant item_ids
        k_values: list of K values to evaluate at
        catalog_size: total catalog size for coverage computation
        item_genre_vectors: optional dict for diversity computation

    Returns:
        Evaluation report dict with per-K metrics and aggregate statistics
    """
    if k_values is None:
        k_values = [5, 10, 20]

    users = list(recommendations_by_user.keys())
    n_users = len(users)

    if n_users == 0:
        return {"error": "No users to evaluate", "n_users": 0}

    results: Dict[str, Any] = {"n_users": n_users, "k_values": k_values}
    all_recs_flat = []

    per_k_metrics: Dict[int, Dict[str, List[float]]] = {
        k: {"ndcg": [], "recall": [], "precision": [], "mrr": [], "ap": []}
        for k in k_values
    }
    mrr_scores = []
    diversity_scores = []

    for user_id in users:
        recs = recommendations_by_user.get(user_id, [])
        relevant = ground_truth_by_user.get(user_id, [])
        if not relevant:
            continue

        all_recs_flat.append(recs)

        for k in k_values:
            per_k_metrics[k]["ndcg"].append(ndcg_at_k(recs, relevant, k))
            per_k_metrics[k]["recall"].append(recall_at_k(recs, relevant, k))
            per_k_metrics[k]["precision"].append(precision_at_k(recs, relevant, k))

        mrr_scores.append(mrr(recs, relevant))
        if item_genre_vectors:
            diversity_scores.append(intra_list_diversity(recs[:k_values[-1]], item_genre_vectors))

    # Aggregate
    for k in k_values:
        for metric_name, scores in per_k_metrics[k].items():
            results[f"{metric_name}@{k}"] = float(np.mean(scores)) if scores else 0.0

    results["mrr"] = float(np.mean(mrr_scores)) if mrr_scores else 0.0

    if catalog_size and all_recs_flat:
        results["coverage"] = coverage(all_recs_flat, catalog_size)

    if diversity_scores:
        results["avg_diversity"] = float(np.mean(diversity_scores))

    # Summary
    logger.info("Evaluation Results:")
    for k in k_values:
        logger.info(
            "  K=%d | NDCG=%.4f | Recall=%.4f | Precision=%.4f",
            k,
            results.get(f"ndcg@{k}", 0),
            results.get(f"recall@{k}", 0),
            results.get(f"precision@{k}", 0),
        )
    logger.info("  MRR=%.4f", results["mrr"])
    if "coverage" in results:
        logger.info("  Coverage=%.4f", results["coverage"])

    return results
