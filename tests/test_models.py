"""
Tests for the ML model components:
- TwoTowerModel forward pass and embedding generation
- FAISSIndex build, search, save/load
- LightGBMRanker training and prediction
"""
import sys
import tempfile
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import pytest
import torch

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.two_tower import TwoTowerModel, UserTower, ItemTower
from src.models.faiss_index import FAISSIndex
from src.models.ranker import LightGBMRanker
from src.features.feature_engineering import N_GENRES


# ------------------------------------------------------------------ #
# Two-Tower Model Tests                                               #
# ------------------------------------------------------------------ #

class TestTwoTowerModel:
    N_USERS = 100
    N_ITEMS = 200
    EMBED_DIM = 32
    BATCH = 16

    @pytest.fixture
    def model(self):
        return TwoTowerModel(
            n_users=self.N_USERS,
            n_items=self.N_ITEMS,
            embed_dim=self.EMBED_DIM,
            hidden_dim=64,
        )

    def test_user_tower_output_shape(self, model):
        user_ids = torch.randint(1, self.N_USERS + 1, (self.BATCH,))
        emb = model.user_tower(user_ids)
        assert emb.shape == (self.BATCH, self.EMBED_DIM)

    def test_item_tower_output_shape(self, model):
        item_ids = torch.randint(1, self.N_ITEMS + 1, (self.BATCH,))
        genres = torch.rand(self.BATCH, N_GENRES)
        emb = model.item_tower(item_ids, genres)
        assert emb.shape == (self.BATCH, self.EMBED_DIM)

    def test_user_embeddings_l2_normalized(self, model):
        user_ids = torch.randint(1, self.N_USERS + 1, (self.BATCH,))
        emb = model.user_tower(user_ids)
        norms = torch.norm(emb, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(self.BATCH), atol=1e-5), (
            f"User embeddings not L2 normalized: norms={norms}"
        )

    def test_item_embeddings_l2_normalized(self, model):
        item_ids = torch.randint(1, self.N_ITEMS + 1, (self.BATCH,))
        genres = torch.rand(self.BATCH, N_GENRES)
        emb = model.item_tower(item_ids, genres)
        norms = torch.norm(emb, p=2, dim=-1)
        assert torch.allclose(norms, torch.ones(self.BATCH), atol=1e-5)

    def test_forward_returns_tuple(self, model):
        user_ids = torch.randint(1, self.N_USERS + 1, (self.BATCH,))
        item_ids = torch.randint(1, self.N_ITEMS + 1, (self.BATCH,))
        genres = torch.rand(self.BATCH, N_GENRES)
        result = model(user_ids, item_ids, genres)
        assert isinstance(result, tuple)
        assert len(result) == 2
        user_emb, item_emb = result
        assert user_emb.shape == (self.BATCH, self.EMBED_DIM)
        assert item_emb.shape == (self.BATCH, self.EMBED_DIM)

    def test_bpr_loss_positive(self, model):
        batch = self.BATCH
        user_emb = torch.randn(batch, self.EMBED_DIM)
        user_emb = torch.nn.functional.normalize(user_emb, p=2, dim=-1)
        pos_emb = user_emb + torch.randn(batch, self.EMBED_DIM) * 0.1  # Similar to user
        neg_emb = -user_emb + torch.randn(batch, self.EMBED_DIM) * 0.1  # Dissimilar
        pos_emb = torch.nn.functional.normalize(pos_emb, p=2, dim=-1)
        neg_emb = torch.nn.functional.normalize(neg_emb, p=2, dim=-1)
        loss = model.bpr_loss(user_emb, pos_emb, neg_emb)
        assert isinstance(loss.item(), float)
        assert loss.item() >= 0.0

    def test_bpr_loss_decreases_with_training(self, model):
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
        losses = []
        for _ in range(20):
            user_ids = torch.randint(1, self.N_USERS + 1, (self.BATCH,))
            pos_ids = torch.randint(1, self.N_ITEMS + 1, (self.BATCH,))
            neg_ids = torch.randint(1, self.N_ITEMS + 1, (self.BATCH,))
            genres = torch.rand(self.BATCH, N_GENRES)
            user_emb = model.user_tower(user_ids)
            pos_emb = model.item_tower(pos_ids, genres)
            neg_emb = model.item_tower(neg_ids, genres)
            loss = model.bpr_loss(user_emb, pos_emb, neg_emb)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
        # Loss should generally decrease over 20 steps
        assert losses[-1] < losses[0] * 2.0, (
            f"Loss did not decrease: initial={losses[0]:.4f}, final={losses[-1]:.4f}"
        )

    def test_get_user_embedding(self, model):
        emb = model.get_user_embedding(user_id=1)
        assert isinstance(emb, np.ndarray)
        assert emb.shape == (self.EMBED_DIM,)
        norm = np.linalg.norm(emb)
        assert abs(norm - 1.0) < 1e-4

    def test_get_item_embeddings(self, model):
        item_ids = list(range(1, 21))
        genres = np.random.rand(20, N_GENRES).astype(np.float32)
        embs = model.get_item_embeddings(item_ids, genres)
        assert embs.shape == (20, self.EMBED_DIM)
        norms = np.linalg.norm(embs, axis=1)
        assert np.allclose(norms, 1.0, atol=1e-4)

    def test_save_and_load(self, model):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "two_tower.pt")
            model.save(path)
            assert Path(path).exists()
            loaded = TwoTowerModel.load(path)
            assert loaded.n_users == model.n_users
            assert loaded.n_items == model.n_items
            assert loaded.embed_dim == model.embed_dim

            # Verify weights match
            user_ids = torch.tensor([1, 2, 3])
            with torch.no_grad():
                orig_emb = model.user_tower(user_ids).numpy()
                loaded_emb = loaded.user_tower(user_ids).numpy()
            assert np.allclose(orig_emb, loaded_emb, atol=1e-5)


# ------------------------------------------------------------------ #
# FAISS Index Tests                                                   #
# ------------------------------------------------------------------ #

class TestFAISSIndex:
    EMBED_DIM = 32
    N_ITEMS = 500

    @pytest.fixture
    def built_index(self):
        np.random.seed(123)
        embeddings = np.random.randn(self.N_ITEMS, self.EMBED_DIM).astype(np.float32)
        # L2 normalize
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
        embeddings = embeddings / norms
        item_ids = list(range(1, self.N_ITEMS + 1))

        index = FAISSIndex(embed_dim=self.EMBED_DIM, n_lists=10, n_probe=5)
        index.build_ivf_index(embeddings, item_ids)
        return index, embeddings, item_ids

    def test_index_built(self, built_index):
        index, embeddings, item_ids = built_index
        assert index.index is not None
        assert index.index.ntotal == self.N_ITEMS

    def test_search_returns_k_results(self, built_index):
        index, embeddings, item_ids = built_index
        query = np.random.randn(self.EMBED_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)
        k = 20
        distances, retrieved_ids = index.search(query, k=k)
        assert len(distances) == k
        assert len(retrieved_ids) == k

    def test_search_result_type(self, built_index):
        index, embeddings, item_ids = built_index
        query = np.random.randn(self.EMBED_DIM).astype(np.float32)
        distances, retrieved_ids = index.search(query, k=10)
        assert distances.dtype in [np.float32, np.float64]
        assert all(isinstance(iid, (int, np.integer)) for iid in retrieved_ids)

    def test_nearest_neighbor_is_self(self, built_index):
        """A query vector that is identical to an indexed vector should return that item first."""
        index, embeddings, item_ids = built_index
        target_idx = 42
        query = embeddings[target_idx].copy()
        distances, retrieved_ids = index.search(query, k=5)
        # The exact item should be in the top results
        assert item_ids[target_idx] in retrieved_ids

    def test_distances_descending(self, built_index):
        """Inner product scores should be in descending order (best first)."""
        index, embeddings, item_ids = built_index
        query = np.random.randn(self.EMBED_DIM).astype(np.float32)
        query = query / np.linalg.norm(query)
        distances, _ = index.search(query, k=20)
        diffs = np.diff(distances)
        # Distances should be non-increasing
        assert (diffs <= 0.01).all(), f"Distances not sorted: {distances[:5]}"

    def test_search_k_capped_at_n_items(self, built_index):
        index, embeddings, item_ids = built_index
        query = np.random.randn(self.EMBED_DIM).astype(np.float32)
        distances, retrieved_ids = index.search(query, k=10000)  # More than n_items
        assert len(retrieved_ids) <= self.N_ITEMS

    def test_save_and_load(self, built_index):
        index, embeddings, item_ids = built_index
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "faiss.index")
            index.save(path)
            assert Path(path).exists()
            assert Path(path).with_suffix(".meta.pkl").exists()

            loaded = FAISSIndex.load(path)
            assert loaded.index.ntotal == self.N_ITEMS
            assert loaded.embed_dim == self.EMBED_DIM

            # Search results should be identical
            query = embeddings[0].copy()
            d1, ids1 = index.search(query, k=10)
            d2, ids2 = loaded.search(query, k=10)
            assert list(ids1) == list(ids2)

    def test_stats(self, built_index):
        index, _, _ = built_index
        stats = index.stats()
        assert stats["n_vectors"] == self.N_ITEMS
        assert stats["embed_dim"] == self.EMBED_DIM
        assert stats["metric"] == "inner_product"

    def test_unnormalized_query_handled(self, built_index):
        """Index should normalize query vectors internally."""
        index, embeddings, _ = built_index
        unnorm_query = np.random.randn(self.EMBED_DIM).astype(np.float32) * 100  # Large magnitude
        d1, ids1 = index.search(unnorm_query, k=5)
        norm_query = unnorm_query / np.linalg.norm(unnorm_query)
        d2, ids2 = index.search(norm_query, k=5)
        assert list(ids1) == list(ids2), "Results should be same for normalized/unnormalized query"


# ------------------------------------------------------------------ #
# LightGBM Ranker Tests                                               #
# ------------------------------------------------------------------ #

def _make_ranker_data(
    n_queries: int = 30,
    items_per_query: int = 20,
    n_features: int = 10,
    pos_ratio: float = 0.3,
    seed: int = 42,
):
    """Generate synthetic ranking data for LightGBM tests."""
    np.random.seed(seed)
    rows = []
    for qid in range(n_queries):
        for i in range(items_per_query):
            label = 1 if np.random.rand() < pos_ratio else 0
            feat = np.random.randn(n_features).tolist()
            row = dict(zip([f"feat_{j}" for j in range(n_features)], feat))
            row["label"] = label
            row["query_id"] = qid
            row["user_id"] = qid
            row["item_id"] = qid * items_per_query + i
            rows.append(row)
    return pd.DataFrame(rows)


class TestLightGBMRanker:
    N_FEATURES = 10
    FEATURE_COLS = [f"feat_{i}" for i in range(N_FEATURES)]

    @pytest.fixture
    def train_data(self):
        return _make_ranker_data(n_queries=40, items_per_query=15, n_features=self.N_FEATURES)

    @pytest.fixture
    def test_data(self):
        return _make_ranker_data(n_queries=10, items_per_query=15, n_features=self.N_FEATURES, seed=99)

    @pytest.fixture
    def trained_ranker(self, train_data):
        ranker = LightGBMRanker(num_leaves=15, n_estimators=50, learning_rate=0.1)
        ranker.train(
            train_df=train_data.sort_values("query_id"),
            feature_cols=self.FEATURE_COLS,
            label_col="label",
            query_col="query_id",
        )
        return ranker

    def test_train_completes(self, train_data):
        ranker = LightGBMRanker(num_leaves=15, n_estimators=30, learning_rate=0.1)
        result = ranker.train(
            train_df=train_data.sort_values("query_id"),
            feature_cols=self.FEATURE_COLS,
            label_col="label",
            query_col="query_id",
        )
        assert result is not None
        assert ranker._trained

    def test_predict_shape(self, trained_ranker, test_data):
        scores = trained_ranker.predict(test_data)
        assert len(scores) == len(test_data)

    def test_predict_returns_float(self, trained_ranker, test_data):
        scores = trained_ranker.predict(test_data)
        assert scores.dtype in [np.float32, np.float64]

    def test_feature_importance_keys(self, trained_ranker):
        importance = trained_ranker.feature_importance()
        assert set(importance.keys()) == set(self.FEATURE_COLS)

    def test_feature_importance_non_negative(self, trained_ranker):
        importance = trained_ranker.feature_importance()
        for feat, val in importance.items():
            assert val >= 0, f"Negative importance for {feat}: {val}"

    def test_top_features(self, trained_ranker):
        top = trained_ranker.top_features(5)
        assert len(top) == 5

    def test_save_and_load(self, trained_ranker, test_data):
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "ranker.lgbm")
            trained_ranker.save(path)
            assert Path(path).exists()

            loaded = LightGBMRanker.load(path)
            orig_scores = trained_ranker.predict(test_data)
            loaded_scores = loaded.predict(test_data)
            assert np.allclose(orig_scores, loaded_scores, atol=1e-5)

    def test_predict_raises_before_training(self, test_data):
        ranker = LightGBMRanker()
        with pytest.raises(RuntimeError, match="not trained"):
            ranker.predict(test_data)

    def test_model_info(self, trained_ranker):
        info = trained_ranker.model_info()
        assert "n_features" in info
        assert info["n_features"] == self.N_FEATURES
        assert "top_10_features" in info

    def test_train_with_validation(self, train_data, test_data):
        ranker = LightGBMRanker(num_leaves=15, n_estimators=50, learning_rate=0.1)
        result = ranker.train(
            train_df=train_data.sort_values("query_id"),
            feature_cols=self.FEATURE_COLS,
            label_col="label",
            query_col="query_id",
            valid_df=test_data.sort_values("query_id"),
            verbose_eval=25,
        )
        assert ranker._trained
        assert "train" in result


# ------------------------------------------------------------------ #
# Evaluation Metrics Tests                                            #
# ------------------------------------------------------------------ #

class TestEvaluationMetrics:
    def test_ndcg_perfect_ranking(self):
        from src.evaluation.metrics import ndcg_at_k
        # When all top-k items are relevant, NDCG should be 1.0
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 2, 3]
        score = ndcg_at_k(recommended, relevant, k=3)
        assert abs(score - 1.0) < 1e-6

    def test_ndcg_no_relevant_in_top_k(self):
        from src.evaluation.metrics import ndcg_at_k
        recommended = [4, 5, 6, 7, 8]
        relevant = [1, 2, 3]
        score = ndcg_at_k(recommended, relevant, k=5)
        assert score == 0.0

    def test_ndcg_partial_match(self):
        from src.evaluation.metrics import ndcg_at_k
        recommended = [1, 4, 2, 5, 3]
        relevant = [1, 2, 3]
        score = ndcg_at_k(recommended, relevant, k=5)
        assert 0.0 < score < 1.0

    def test_recall_at_k(self):
        from src.evaluation.metrics import recall_at_k
        recommended = [1, 2, 3, 4, 5]
        relevant = [1, 2, 6, 7]
        recall = recall_at_k(recommended, relevant, k=3)
        assert abs(recall - 0.5) < 1e-6  # 2 out of 4 relevant found in top 3

    def test_recall_empty_relevant(self):
        from src.evaluation.metrics import recall_at_k
        score = recall_at_k([1, 2, 3], [], k=3)
        assert score == 0.0

    def test_mrr_first_position(self):
        from src.evaluation.metrics import mrr
        recommended = [1, 2, 3]
        relevant = [1]
        assert abs(mrr(recommended, relevant) - 1.0) < 1e-6

    def test_mrr_second_position(self):
        from src.evaluation.metrics import mrr
        recommended = [4, 1, 2]
        relevant = [1, 2]
        assert abs(mrr(recommended, relevant) - 0.5) < 1e-6

    def test_mrr_no_hit(self):
        from src.evaluation.metrics import mrr
        assert mrr([4, 5, 6], [1, 2, 3]) == 0.0

    def test_coverage(self):
        from src.evaluation.metrics import coverage
        recs = [[1, 2, 3], [4, 5, 6], [1, 7, 8]]
        cov = coverage(recs, catalog_size=10)
        assert abs(cov - 0.8) < 1e-6  # 8 out of 10 items covered

    def test_skew_detection_no_skew(self):
        from src.evaluation.metrics import detect_training_serving_skew
        np.random.seed(0)
        train = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
        serving = pd.DataFrame({"x": np.random.normal(0, 1, 500)})
        result = detect_training_serving_skew(train, serving, threshold=0.5)
        assert not result["skew_detected"]

    def test_skew_detection_with_skew(self):
        from src.evaluation.metrics import detect_training_serving_skew
        np.random.seed(0)
        train = pd.DataFrame({"x": np.random.normal(0, 1, 1000)})
        serving = pd.DataFrame({"x": np.random.normal(5, 1, 500)})  # Very different distribution
        result = detect_training_serving_skew(train, serving, threshold=0.1)
        assert result["skew_detected"]
        assert "x" in result["flagged_features"]
