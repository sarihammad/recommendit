"""
Training script for the LightGBM LambdaMART ranker.
Builds proper LTR training pairs from MovieLens interactions with
user+item+interaction features, evaluates on a holdout set.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from src.config import settings
from src.evaluation.metrics import ndcg_at_k, recall_at_k
from src.features.feature_engineering import FeatureEngineer
from src.models.ranker import LightGBMRanker

logger = logging.getLogger(__name__)


class RankerTrainer:
    """
    Orchestrates feature building, pair construction, LTR training,
    and evaluation for the LightGBM ranker.
    """

    def __init__(
        self,
        data_dir: str = None,
        model_output_path: str = None,
        features_dir: str = "data/features",
        n_negatives: int = None,
        num_leaves: int = None,
        n_estimators: int = None,
        learning_rate: float = None,
    ):
        self.data_dir = data_dir or settings.DATA_DIR
        self.model_output_path = model_output_path or settings.RANKER_MODEL_PATH
        self.features_dir = features_dir
        self.n_negatives = n_negatives or settings.N_NEGATIVES
        self.num_leaves = num_leaves or settings.LGBM_NUM_LEAVES
        self.n_estimators = n_estimators or settings.LGBM_N_ESTIMATORS
        self.learning_rate = learning_rate or settings.LGBM_LEARNING_RATE

    def run(self) -> LightGBMRanker:
        """
        Full training pipeline:
        1. Load/compute features
        2. Build training pairs
        3. Attach interaction features
        4. Train LambdaMART
        5. Evaluate on holdout
        6. Save model

        Returns:
            Trained LightGBMRanker
        """
        fe = FeatureEngineer(self.data_dir)
        fe.load_data()

        # Try loading pre-built features; otherwise compute fresh
        feat_dir = Path(self.features_dir)
        user_feat_path = feat_dir / "user_features.parquet"
        item_feat_path = feat_dir / "item_features.parquet"

        if user_feat_path.exists() and item_feat_path.exists():
            logger.info("Loading pre-built features from %s ...", self.features_dir)
            fe.load_features(self.features_dir)
        else:
            logger.info("Computing features from scratch ...")
            fe.build_user_features()
            fe.build_item_features()
            fe.save_features(self.features_dir)

        # Build training pairs (positive + negative interactions)
        train_pairs, test_pairs = fe.build_training_pairs(
            ratings_df=fe.ratings_df,
            n_negatives=self.n_negatives,
        )

        # Attach interaction + contextual features
        logger.info("Building interaction features for training set (%d pairs)...", len(train_pairs))
        train_feats = fe.build_interaction_features(train_pairs)

        logger.info("Building interaction features for test set (%d pairs)...", len(test_pairs))
        test_feats = fe.build_interaction_features(test_pairs)

        feature_cols = fe.get_feature_columns()
        # Only keep columns that exist in the feature DataFrame
        feature_cols = [c for c in feature_cols if c in train_feats.columns]

        logger.info("Feature columns: %d features", len(feature_cols))

        # Drop rows with any NaN in feature columns
        train_feats = train_feats.dropna(subset=feature_cols).reset_index(drop=True)
        test_feats = test_feats.dropna(subset=feature_cols).reset_index(drop=True)

        # Sort by query_id for proper LightGBM group assignment
        train_feats = train_feats.sort_values("query_id").reset_index(drop=True)
        test_feats = test_feats.sort_values("query_id").reset_index(drop=True)

        logger.info(
            "Training set: %d samples, %d queries",
            len(train_feats), train_feats["query_id"].nunique(),
        )
        logger.info(
            "Test set: %d samples, %d queries",
            len(test_feats), test_feats["query_id"].nunique(),
        )

        # Train ranker
        ranker = LightGBMRanker(
            num_leaves=self.num_leaves,
            n_estimators=self.n_estimators,
            learning_rate=self.learning_rate,
        )

        evals_result = ranker.train(
            train_df=train_feats,
            feature_cols=feature_cols,
            label_col="label",
            query_col="query_id",
            valid_df=test_feats,
            verbose_eval=50,
        )

        # Evaluate on holdout
        self._evaluate_holdout(ranker, test_feats, feature_cols)

        # Save model
        ranker.save(self.model_output_path)

        # Log feature importance
        top_feats = ranker.top_features(10)
        logger.info("Top 10 features by gain: %s", top_feats)

        return ranker

    def _evaluate_holdout(
        self,
        ranker: LightGBMRanker,
        test_df: pd.DataFrame,
        feature_cols: List[str],
    ) -> Dict[str, float]:
        """Evaluate ranker on holdout set and log NDCG/Recall metrics."""
        scores = ranker.predict(test_df)
        test_df = test_df.copy()
        test_df["pred_score"] = scores

        ndcg_10_list = []
        ndcg_20_list = []
        recall_20_list = []

        for query_id, group in test_df.groupby("query_id"):
            group_sorted = group.sort_values("pred_score", ascending=False)
            recommended = group_sorted["item_id"].tolist()
            relevant = group[group["label"] == 1]["item_id"].tolist()
            if not relevant:
                continue
            ndcg_10_list.append(ndcg_at_k(recommended, relevant, k=10))
            ndcg_20_list.append(ndcg_at_k(recommended, relevant, k=20))
            recall_20_list.append(recall_at_k(recommended, relevant, k=20))

        metrics = {
            "ndcg@10": float(np.mean(ndcg_10_list)),
            "ndcg@20": float(np.mean(ndcg_20_list)),
            "recall@20": float(np.mean(recall_20_list)),
            "n_queries": len(ndcg_10_list),
        }

        logger.info("Holdout evaluation results:")
        for k, v in metrics.items():
            logger.info("  %s: %.4f", k, v)

        return metrics
