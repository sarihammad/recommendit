"""Item-item collaborative filtering based on co-occurrence and cosine similarity."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import structlog
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cosine
from sklearn.metrics.pairwise import cosine_similarity
import pickle
from pathlib import Path
from ..service.config import config

logger = structlog.get_logger(__name__)

class ItemItemRecommender:
    """Item-item collaborative filtering recommender."""
    
    def __init__(self, domain: str):
        """Initialize item-item recommender."""
        self.domain = domain
        self.item_similarity_matrix = None
        self.item_mapping = {}
        self.reverse_item_mapping = {}
        self.user_item_matrix = None
        self.user_mapping = {}
        self.reverse_user_mapping = {}
        
    def fit(self, interactions_df: pd.DataFrame):
        """Fit item-item model on interactions data."""
        logger.info("Fitting item-item model", domain=self.domain)
        
        # Create mappings
        self._create_mappings(interactions_df)
        
        # Create user-item matrix
        self.user_item_matrix = self._create_user_item_matrix(interactions_df)
        
        # Compute item-item similarities
        self._compute_item_similarities()
        
        logger.info("Item-item model fitted", 
                   domain=self.domain,
                   users=len(self.user_mapping),
                   items=len(self.item_mapping))
    
    def _create_mappings(self, interactions_df: pd.DataFrame):
        """Create user and item ID mappings."""
        # Get unique users and items
        unique_users = sorted(interactions_df['user_id'].unique())
        unique_items = sorted(interactions_df['item_id'].unique())
        
        # Create mappings
        self.user_mapping = {user_id: idx for idx, user_id in enumerate(unique_users)}
        self.item_mapping = {item_id: idx for idx, item_id in enumerate(unique_items)}
        
        # Create reverse mappings
        self.reverse_user_mapping = {idx: user_id for user_id, idx in self.user_mapping.items()}
        self.reverse_item_mapping = {idx: item_id for item_id, idx in self.item_mapping.items()}
        
        logger.info("Mappings created", 
                   domain=self.domain,
                   num_users=len(self.user_mapping),
                   num_items=len(self.item_mapping))
    
    def _create_user_item_matrix(self, interactions_df: pd.DataFrame) -> csr_matrix:
        """Create sparse user-item matrix."""
        # Map user and item IDs to indices
        user_indices = [self.user_mapping[user_id] for user_id in interactions_df['user_id']]
        item_indices = [self.item_mapping[item_id] for item_id in interactions_df['item_id']]
        weights = interactions_df['weight'].values
        
        # Create sparse matrix
        matrix = csr_matrix(
            (weights, (user_indices, item_indices)),
            shape=(len(self.user_mapping), len(self.item_mapping))
        )
        
        return matrix
    
    def _compute_item_similarities(self):
        """Compute item-item similarity matrix."""
        logger.info("Computing item similarities", domain=self.domain)
        
        # Convert to dense matrix for similarity computation
        # Note: This might be memory intensive for large datasets
        # In production, consider using sparse similarity computation
        dense_matrix = self.user_item_matrix.toarray()
        
        # Compute cosine similarity
        self.item_similarity_matrix = cosine_similarity(dense_matrix.T)
        
        # Set diagonal to 0 (items are not similar to themselves)
        np.fill_diagonal(self.item_similarity_matrix, 0)
        
        logger.info("Item similarities computed", 
                   domain=self.domain,
                   shape=self.item_similarity_matrix.shape)
    
    def get_similar_items(self, item_id: str, n_similar: int = 20) -> List[Dict[str, Any]]:
        """Get similar items for a given item."""
        if self.item_similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if item_id not in self.item_mapping:
            logger.warning("Item not in training set", item_id=item_id, domain=self.domain)
            return []
        
        # Get item index
        item_idx = self.item_mapping[item_id]
        
        # Get similarities for this item
        item_similarities = self.item_similarity_matrix[item_idx]
        
        # Get top similar items
        top_indices = np.argsort(item_similarities)[::-1][:n_similar]
        
        # Convert to list
        candidates = []
        for idx in top_indices:
            if item_similarities[idx] > 0:  # Only include positive similarities
                similar_item_id = self.reverse_item_mapping[idx]
                candidates.append({
                    'item_id': similar_item_id,
                    'score': float(item_similarities[idx]),
                    'source': 'item_item'
                })
        
        return candidates
    
    def get_user_candidates_from_history(self, user_id: str, user_history: List[str], 
                                       n_candidates: int = 100) -> List[Dict[str, Any]]:
        """Get candidates based on user's interaction history."""
        if self.item_similarity_matrix is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if not user_history:
            return []
        
        # Get indices of items in user history
        history_indices = []
        for item_id in user_history:
            if item_id in self.item_mapping:
                history_indices.append(self.item_mapping[item_id])
        
        if not history_indices:
            return []
        
        # Aggregate similarities from all items in history
        aggregated_similarities = np.zeros(len(self.item_mapping))
        
        for item_idx in history_indices:
            item_similarities = self.item_similarity_matrix[item_idx]
            aggregated_similarities += item_similarities
        
        # Average the similarities
        aggregated_similarities /= len(history_indices)
        
        # Remove items already in history
        for item_idx in history_indices:
            aggregated_similarities[item_idx] = 0
        
        # Get top candidates
        top_indices = np.argsort(aggregated_similarities)[::-1][:n_candidates]
        
        # Convert to list
        candidates = []
        for idx in top_indices:
            if aggregated_similarities[idx] > 0:
                item_id = self.reverse_item_mapping[idx]
                candidates.append({
                    'item_id': item_id,
                    'score': float(aggregated_similarities[idx]),
                    'source': 'item_item_history'
                })
        
        return candidates
    
    def get_user_candidates(self, user_id: str, n_candidates: int = 100) -> List[Dict[str, Any]]:
        """Get candidates for a user based on their interaction history."""
        if user_id not in self.user_mapping:
            logger.warning("User not in training set", user_id=user_id, domain=self.domain)
            return []
        
        # Get user's interaction history
        user_idx = self.user_mapping[user_id]
        user_interactions = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Get items the user has interacted with
        user_items = []
        for item_idx, weight in enumerate(user_interactions):
            if weight > 0:
                item_id = self.reverse_item_mapping[item_idx]
                user_items.append(item_id)
        
        # Get candidates based on history
        return self.get_user_candidates_from_history(user_id, user_items, n_candidates)
    
    def predict_score(self, user_id: str, item_id: str) -> float:
        """Predict interaction score for user-item pair."""
        if self.item_similarity_matrix is None:
            return 0.0
        
        if user_id not in self.user_mapping or item_id not in self.item_mapping:
            return 0.0
        
        # Get user's interaction history
        user_idx = self.user_mapping[user_id]
        item_idx = self.item_mapping[item_id]
        
        user_interactions = self.user_item_matrix[user_idx].toarray().flatten()
        
        # Get similarities between target item and user's history
        item_similarities = self.item_similarity_matrix[item_idx]
        
        # Weighted average of similarities based on user's interaction weights
        weighted_similarity = np.sum(user_interactions * item_similarities)
        total_weight = np.sum(user_interactions)
        
        if total_weight > 0:
            return float(weighted_similarity / total_weight)
        else:
            return 0.0
    
    def get_item_similarity(self, item_id_1: str, item_id_2: str) -> float:
        """Get similarity between two items."""
        if self.item_similarity_matrix is None:
            return 0.0
        
        if item_id_1 not in self.item_mapping or item_id_2 not in self.item_mapping:
            return 0.0
        
        idx_1 = self.item_mapping[item_id_1]
        idx_2 = self.item_mapping[item_id_2]
        
        return float(self.item_similarity_matrix[idx_1, idx_2])
    
    def save(self, domain: str):
        """Save item-item model and mappings."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save similarity matrix
        if self.item_similarity_matrix is not None:
            np.save(artifacts_dir / "item_similarity_matrix.npy", self.item_similarity_matrix)
        
        # Save mappings
        mappings = {
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping
        }
        
        with open(artifacts_dir / "item_item_mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)
        
        # Save user-item matrix
        if self.user_item_matrix is not None:
            with open(artifacts_dir / "user_item_matrix.pkl", "wb") as f:
                pickle.dump(self.user_item_matrix, f)
        
        logger.info("Item-item model saved", domain=domain)
    
    def load(self, domain: str):
        """Load item-item model and mappings."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        
        # Load similarity matrix
        similarity_path = artifacts_dir / "item_similarity_matrix.npy"
        if similarity_path.exists():
            self.item_similarity_matrix = np.load(similarity_path)
        
        # Load mappings
        mappings_path = artifacts_dir / "item_item_mappings.pkl"
        if mappings_path.exists():
            with open(mappings_path, "rb") as f:
                mappings = pickle.load(f)
                self.user_mapping = mappings['user_mapping']
                self.item_mapping = mappings['item_mapping']
                self.reverse_user_mapping = mappings['reverse_user_mapping']
                self.reverse_item_mapping = mappings['reverse_item_mapping']
        
        # Load user-item matrix
        matrix_path = artifacts_dir / "user_item_matrix.pkl"
        if matrix_path.exists():
            with open(matrix_path, "rb") as f:
                self.user_item_matrix = pickle.load(f)
        
        logger.info("Item-item model loaded", domain=domain) 