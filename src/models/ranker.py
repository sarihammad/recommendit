"""
LightGBM LambdaMART ranker for re-ranking FAISS candidates.
Uses Learning-to-Rank with query groups for proper LTR training.
"""
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import lightgbm as lgb
    LGB_AVAILABLE = True
except ImportError:
    LGB_AVAILABLE = False
    logger_tmp = logging.getLogger(__name__)
    logger_tmp.warning("lightgbm not available")

logger = logging.getLogger(__name__)


class LightGBMRanker:
    """
    LightGBM LambdaMART learning-to-rank model.

    Trains with query groups so the model learns to rank items
    within each user's candidate set, rather than doing pointwise prediction.
    """

    def __init__(
        self,
        num_leaves: int = 63,
        n_estimators: int = 500,
        learning_rate: float = 0.05,
        eval_at: List[int] = None,
    ):
        if not LGB_AVAILABLE:
            raise ImportError("lightgbm is required. Install with: pip install lightgbm")
        self.num_leaves = num_leaves
        self.n_estimators = n_estimators
        self.learning_rate = learning_rate
        self.eval_at = eval_at or [5, 10, 20]
        self.model: Optional[lgb.Booster] = None
        self.feature_names: Optional[List[str]] = None
        self._trained = False

    # ------------------------------------------------------------------ #
    # Training                                                             #
    # ------------------------------------------------------------------ #

    def train(
        self,
        train_df: pd.DataFrame,
        feature_cols: List[str],
        label_col: str = "label",
        query_col: str = "query_id",
        valid_df: Optional[pd.DataFrame] = None,
        verbose_eval: int = 50,
    ) -> Dict[str, List[float]]:
        """
        Train the LambdaMART ranker.

        Args:
            train_df: DataFrame with features, labels, and query group IDs
            feature_cols: columns to use as ranking features
            label_col: binary or graded relevance label column
            query_col: column identifying query groups (user_id groups)
            valid_df: optional validation set with same structure
            verbose_eval: log evaluation every N rounds

        Returns:
            evals_result dict containing training/validation metrics history
        """
        self.feature_names = feature_cols
        X_train = train_df[feature_cols].values.astype(np.float32)
        y_train = train_df[label_col].values.astype(np.float32)

        # Compute group sizes (how many items per query)
        train_groups = (
            train_df.groupby(query_col, sort=False)
            .size()
            .values
        )

        lgb_train = lgb.Dataset(
            X_train,
            label=y_train,
            group=train_groups,
            feature_name=feature_cols,
            free_raw_data=False,
        )

        valid_sets = [lgb_train]
        valid_names = ["train"]
        if valid_df is not None:
            X_valid = valid_df[feature_cols].values.astype(np.float32)
            y_valid = valid_df[label_col].values.astype(np.float32)
            valid_groups = (
                valid_df.groupby(query_col, sort=False)
                .size()
                .values
            )
            lgb_valid = lgb.Dataset(
                X_valid,
                label=y_valid,
                group=valid_groups,
                feature_name=feature_cols,
                reference=lgb_train,
                free_raw_data=False,
            )
            valid_sets.append(lgb_valid)
            valid_names.append("valid")

        params = {
            "objective": "lambdarank",
            "metric": "ndcg",
            "eval_at": self.eval_at,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "min_child_samples": 20,
            "subsample": 0.8,
            "colsample_bytree": 0.8,
            "reg_alpha": 0.1,
            "reg_lambda": 0.1,
            "label_gain": [0, 1, 3, 7, 15],  # Relevance gain mapping
            "verbose": -1,
            "n_jobs": -1,
        }

        evals_result: Dict = {}
        callbacks = [
            lgb.log_evaluation(period=verbose_eval),
            lgb.record_evaluation(evals_result),
        ]
        if valid_df is not None:
            callbacks.append(lgb.early_stopping(stopping_rounds=30, verbose=True))

        logger.info(
            "Training LambdaMART ranker: %d samples, %d features, %d queries",
            len(train_df), len(feature_cols), len(train_groups),
        )

        self.model = lgb.train(
            params,
            lgb_train,
            num_boost_round=self.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        self._trained = True

        logger.info("Training complete. Best iteration: %d", self.model.best_iteration)
        return evals_result

    # ------------------------------------------------------------------ #
    # Inference                                                            #
    # ------------------------------------------------------------------ #

    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """
        Score items for ranking.

        Args:
            features_df: DataFrame with the same feature columns used during training

        Returns:
            Array of scores (higher = more relevant)
        """
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained. Call train() first.")
        X = features_df[self.feature_names].values.astype(np.float32)
        return self.model.predict(X)

    # ------------------------------------------------------------------ #
    # Analysis                                                             #
    # ------------------------------------------------------------------ #

    def feature_importance(self, importance_type: str = "gain") -> Dict[str, float]:
        """
        Return feature importances as a dict.

        Args:
            importance_type: 'gain' (default) or 'split'
        """
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained.")
        importances = self.model.feature_importance(importance_type=importance_type)
        names = self.model.feature_name()
        result = dict(zip(names, importances.tolist()))
        return dict(sorted(result.items(), key=lambda x: x[1], reverse=True))

    def top_features(self, n: int = 10, importance_type: str = "gain") -> Dict[str, float]:
        """Return the top-N most important features."""
        imp = self.feature_importance(importance_type)
        return dict(list(imp.items())[:n])

    # ------------------------------------------------------------------ #
    # Persistence                                                          #
    # ------------------------------------------------------------------ #

    def save(self, path: str) -> None:
        """Save the trained model to disk."""
        if not self._trained or self.model is None:
            raise RuntimeError("Model not trained.")
        save_path = Path(path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save_model(str(save_path))
        logger.info("Saved LightGBM ranker to %s", save_path)

    @classmethod
    def load(cls, path: str) -> "LightGBMRanker":
        """Load a saved model from disk."""
        load_path = Path(path)
        if not load_path.exists():
            raise FileNotFoundError(f"Ranker model not found at {load_path}")
        obj = cls()
        obj.model = lgb.Booster(model_file=str(load_path))
        obj.feature_names = obj.model.feature_name()
        obj._trained = True
        logger.info(
            "Loaded LightGBM ranker from %s (%d features, %d trees)",
            load_path, len(obj.feature_names), obj.model.num_trees(),
        )
        return obj

    @property
    def n_features(self) -> int:
        return len(self.feature_names) if self.feature_names else 0

    @property
    def best_iteration(self) -> int:
        if self.model is not None:
            return self.model.best_iteration
        return 0

    def model_info(self) -> Dict:
        """Return model metadata."""
        if not self._trained:
            return {"status": "not trained"}
        return {
            "n_features": self.n_features,
            "best_iteration": self.best_iteration,
            "num_leaves": self.num_leaves,
            "learning_rate": self.learning_rate,
            "eval_at": self.eval_at,
            "top_10_features": self.top_features(10) if self._trained else {},
        }
