"""
End-to-end pipeline orchestrator for RecommendIt.
CLI entry point for running individual stages or the full training pipeline.
"""
import argparse
import logging
import sys
import time
from pathlib import Path
from typing import List, Optional

from src.config import settings

logging.basicConfig(
    level=getattr(logging, settings.LOG_LEVEL),
    format="%(asctime)s  %(levelname)-8s  %(name)s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)

VALID_STAGES = ["all", "data", "features", "embeddings", "index", "ranker", "evaluate", "load_features"]


class PipelineOrchestrator:
    """
    Orchestrates the full RecommendIt training pipeline.
    Each stage can be run independently or all stages can be run sequentially.
    """

    def __init__(
        self,
        data_dir: str = None,
        models_dir: str = "models",
        features_dir: str = "data/features",
    ):
        self.data_dir = data_dir or settings.DATA_DIR
        self.models_dir = Path(models_dir)
        self.features_dir = features_dir
        self.models_dir.mkdir(parents=True, exist_ok=True)

    def _timed(self, stage_name: str, fn, *args, **kwargs):
        """Run a stage function, logging elapsed time."""
        logger.info("=" * 60)
        logger.info("STAGE: %s", stage_name.upper())
        logger.info("=" * 60)
        t0 = time.time()
        result = fn(*args, **kwargs)
        elapsed = time.time() - t0
        logger.info("STAGE %s completed in %.2fs", stage_name.upper(), elapsed)
        return result

    # ------------------------------------------------------------------ #
    # Individual Stages                                                    #
    # ------------------------------------------------------------------ #

    def run_data_download(self) -> None:
        """Download MovieLens 1M dataset if not already present."""
        from data.download import download_movielens
        data_parent = str(Path(self.data_dir).parent)
        download_movielens(output_dir=data_parent)

    def run_features(self) -> None:
        """Build and save user/item feature DataFrames."""
        from src.features.feature_engineering import FeatureEngineer
        fe = FeatureEngineer(self.data_dir)
        fe.load_data()
        fe.build_user_features()
        fe.build_item_features()
        fe.save_features(self.features_dir)
        logger.info("Features saved to %s", self.features_dir)

    def run_load_features(self) -> None:
        """Load features into Redis feature store."""
        import pandas as pd
        from src.features.feature_store import RedisFeatureStore

        feat_dir = Path(self.features_dir)
        user_feat_path = feat_dir / "user_features.parquet"
        item_feat_path = feat_dir / "item_features.parquet"

        if not user_feat_path.exists() or not item_feat_path.exists():
            logger.warning("Feature files not found. Running feature engineering first...")
            self.run_features()

        user_features_df = pd.read_parquet(user_feat_path)
        item_features_df = pd.read_parquet(item_feat_path)

        store = RedisFeatureStore(redis_url=settings.REDIS_URL)
        store.load_all_features(user_features_df, item_features_df)
        logger.info("Features loaded into feature store. Stats: %s", store.stats())

    def run_embeddings(self) -> None:
        """Train the two-tower embedding model."""
        from src.training.train_embeddings import EmbeddingTrainer
        trainer = EmbeddingTrainer(
            data_dir=self.data_dir,
            model_output_path=str(self.models_dir / "two_tower.pt"),
        )
        trainer.train()

    def run_index(self) -> None:
        """Build the FAISS ANN index from trained embeddings."""
        from src.training.build_index import IndexBuilder
        builder = IndexBuilder(
            model_path=str(self.models_dir / "two_tower.pt"),
            index_output_path=str(self.models_dir / "faiss.index"),
            data_dir=self.data_dir,
        )
        builder.build()

    def run_ranker(self) -> None:
        """Train the LightGBM LambdaMART ranker."""
        from src.training.train_ranker import RankerTrainer
        trainer = RankerTrainer(
            data_dir=self.data_dir,
            model_output_path=str(self.models_dir / "ranker.lgbm"),
            features_dir=self.features_dir,
        )
        trainer.run()

    def run_evaluate(self) -> None:
        """
        Run offline evaluation on the trained pipeline.
        Loads test interactions from ratings data and computes NDCG/Recall/MRR.
        """
        import numpy as np
        import pandas as pd
        from src.evaluation.metrics import evaluate_model
        from src.features.feature_engineering import FeatureEngineer
        from src.models.faiss_index import FAISSIndex
        from src.models.ranker import LightGBMRanker
        from src.models.two_tower import TwoTowerModel
        import torch

        model_path = str(self.models_dir / "two_tower.pt")
        index_path = str(self.models_dir / "faiss.index")
        ranker_path = str(self.models_dir / "ranker.lgbm")

        for path in [model_path, index_path, ranker_path]:
            if not Path(path).exists():
                logger.error("Required model file not found: %s", path)
                logger.error("Run the full pipeline first with: --stage all")
                return

        device = torch.device("cpu")
        model = TwoTowerModel.load(model_path, device=device)
        faiss_index = FAISSIndex.load(index_path)
        ranker = LightGBMRanker.load(ranker_path)

        fe = FeatureEngineer(self.data_dir)
        fe.load_data()

        # Use last 10% of interactions per user as test set
        ratings = fe.ratings_df.sort_values("timestamp")
        test_ratings = ratings.groupby("user_id").tail(
            max(1, int(len(ratings) * 0.1 / ratings["user_id"].nunique()))
        )

        # Sample a subset of users for evaluation (limit for speed)
        eval_users = test_ratings["user_id"].unique()[:200]
        logger.info("Evaluating on %d users...", len(eval_users))

        ground_truth = {}
        recommendations = {}

        from src.features.feature_store import RedisFeatureStore
        from src.features.feature_engineering import N_GENRES, GENRE_TO_IDX
        store = RedisFeatureStore(redis_url=settings.REDIS_URL)

        for user_id in eval_users:
            ground_truth[user_id] = test_ratings[
                (test_ratings["user_id"] == user_id) & (test_ratings["rating"] >= 4)
            ]["item_id"].tolist()
            if not ground_truth[user_id]:
                continue

            user_emb = model.get_user_embedding(int(user_id), device=device)
            distances, candidate_ids = faiss_index.search(user_emb, k=settings.TOP_K_CANDIDATES)

            if len(candidate_ids) == 0:
                continue

            user_feat = store.get_user_features(int(user_id)) or {}
            item_feats = store.get_item_features_batch(candidate_ids.tolist())

            from src.serving.recommender import RecommendationPipeline
            # Build feature rows for ranker
            rows = []
            for iid in candidate_ids.tolist():
                ifeat = item_feats.get(iid) or {}
                row = {
                    "item_id": iid,
                    "avg_rating": float(user_feat.get("avg_rating", 3.5)),
                    "log_rating_count": float(user_feat.get("log_rating_count", 0.0)),
                    "recency_score": float(user_feat.get("recency_score", 0.5)),
                    "gender_encoded": float(user_feat.get("gender_encoded", 0.0)),
                    "age_normalized": float(user_feat.get("age_normalized", 0.3)),
                    "occupation_normalized": float(user_feat.get("occupation_normalized", 0.3)),
                    "item_avg_rating": float(ifeat.get("avg_rating", 3.5)),
                    "item_log_rating_count": float(ifeat.get("log_rating_count", 0.0)),
                    "popularity_score": float(ifeat.get("popularity_score", 0.0)),
                    "rating_stddev": float(ifeat.get("rating_stddev", 0.0)),
                    "year_normalized": float(ifeat.get("year_normalized", 0.5)),
                }
                row["rating_diff"] = row["avg_rating"] - row["item_avg_rating"]
                row["user_item_popularity_ratio"] = row["log_rating_count"] / (row["item_log_rating_count"] + 1e-8)
                ugp = user_feat.get("genre_pref", [0.0] * N_GENRES)
                igv = ifeat.get("genre_vector", [0.0] * N_GENRES)
                for i in range(N_GENRES):
                    row[f"user_genre_{i}"] = float(ugp[i]) if i < len(ugp) else 0.0
                    row[f"item_genre_{i}"] = float(igv[i]) if i < len(igv) else 0.0
                row["genre_affinity"] = sum(row[f"user_genre_{i}"] * row[f"item_genre_{i}"] for i in range(N_GENRES))
                rows.append(row)

            import pandas as pd
            feat_df = pd.DataFrame(rows).fillna(0.0)
            for col in ranker.feature_names:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            scores = ranker.predict(feat_df)
            feat_df["score"] = scores
            ranked_items = feat_df.nlargest(20, "score")["item_id"].tolist()
            recommendations[user_id] = ranked_items

        results = evaluate_model(
            recommendations_by_user=recommendations,
            ground_truth_by_user=ground_truth,
            k_values=[5, 10, 20],
            catalog_size=fe.movies_df["item_id"].nunique(),
        )

        logger.info("Evaluation Report:")
        for k, v in results.items():
            if isinstance(v, float):
                logger.info("  %s: %.4f", k, v)
            else:
                logger.info("  %s: %s", k, v)

    # ------------------------------------------------------------------ #
    # Full Pipeline                                                        #
    # ------------------------------------------------------------------ #

    def run_all(self) -> None:
        """Run the complete training pipeline end-to-end."""
        stages = [
            ("Data Download", self.run_data_download),
            ("Feature Engineering", self.run_features),
            ("Train Embeddings", self.run_embeddings),
            ("Build FAISS Index", self.run_index),
            ("Train Ranker", self.run_ranker),
            ("Load Features to Store", self.run_load_features),
            ("Evaluate", self.run_evaluate),
        ]

        total_start = time.time()
        for stage_name, fn in stages:
            try:
                self._timed(stage_name, fn)
            except Exception as exc:
                logger.error("Stage '%s' failed: %s", stage_name, exc, exc_info=True)
                logger.error("Aborting pipeline.")
                sys.exit(1)

        total_elapsed = time.time() - total_start
        logger.info("=" * 60)
        logger.info("Full pipeline completed in %.2fs", total_elapsed)
        logger.info("=" * 60)

    def run_stage(self, stage: str) -> None:
        stage_map = {
            "data": self.run_data_download,
            "features": self.run_features,
            "load_features": self.run_load_features,
            "embeddings": self.run_embeddings,
            "index": self.run_index,
            "ranker": self.run_ranker,
            "evaluate": self.run_evaluate,
            "all": self.run_all,
        }
        fn = stage_map.get(stage)
        if fn is None:
            logger.error("Unknown stage: %s. Valid stages: %s", stage, VALID_STAGES)
            sys.exit(1)
        if stage == "all":
            fn()
        else:
            self._timed(stage, fn)


def main():
    parser = argparse.ArgumentParser(
        description="RecommendIt Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Stages:
  all             Run the complete pipeline end-to-end
  data            Download MovieLens 1M dataset
  features        Build user/item feature DataFrames
  load_features   Load features into Redis feature store
  embeddings      Train two-tower embedding model
  index           Build FAISS ANN index
  ranker          Train LightGBM LambdaMART ranker
  evaluate        Run offline evaluation

Examples:
  python -m src.pipelines.run_pipeline --stage all
  python -m src.pipelines.run_pipeline --stage embeddings --models-dir models/
  python -m src.pipelines.run_pipeline --stage evaluate
        """,
    )
    parser.add_argument(
        "--stage",
        type=str,
        default="all",
        choices=VALID_STAGES,
        help="Pipeline stage to run (default: all)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default=settings.DATA_DIR,
        help="Directory containing MovieLens data",
    )
    parser.add_argument(
        "--models-dir",
        type=str,
        default="models",
        help="Directory to save/load model artifacts",
    )
    parser.add_argument(
        "--features-dir",
        type=str,
        default="data/features",
        help="Directory to save/load feature parquet files",
    )

    args = parser.parse_args()

    orchestrator = PipelineOrchestrator(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        features_dir=args.features_dir,
    )
    orchestrator.run_stage(args.stage)


if __name__ == "__main__":
    main()
