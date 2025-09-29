"""ALS (Alternating Least Squares) collaborative filtering model."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import structlog
from implicit.als import AlternatingLeastSquares
from scipy.sparse import csr_matrix
import pickle
from pathlib import Path
from ..service.config import config

logger = structlog.get_logger(__name__)

class ALSRecommender:
    """ALS-based collaborative filtering recommender."""
    
    def __init__(self, domain: str):
        """Initialize ALS recommender."""
        self.domain = domain
        self.model = None
        self.user_mapping = {}
        self.item_mapping = {}
        self.reverse_user_mapping = {}
        self.reverse_item_mapping = {}
        self.user_factors = None
        self.item_factors = None
        
    def fit(self, interactions_df: pd.DataFrame):
        """Fit ALS model on interactions data."""
        logger.info("Fitting ALS model", domain=self.domain)
        
        # Create user and item mappings
        self._create_mappings(interactions_df)
        
        # Create interaction matrix
        interaction_matrix = self._create_interaction_matrix(interactions_df)
        
        # Initialize and fit ALS model
        self.model = AlternatingLeastSquares(
            factors=config.ALS_FACTORS,
            iterations=config.ALS_ITERATIONS,
            regularization=config.ALS_REGULARIZATION,
            alpha=config.ALS_ALPHA,
            random_state=42
        )
        
        # Fit the model
        self.model.fit(interaction_matrix)
        
        # Store factors
        self.user_factors = self.model.user_factors
        self.item_factors = self.model.item_factors
        
        logger.info("ALS model fitted", 
                   domain=self.domain,
                   users=len(self.user_mapping),
                   items=len(self.item_mapping),
                   factors=config.ALS_FACTORS)
    
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
    
    def _create_interaction_matrix(self, interactions_df: pd.DataFrame) -> csr_matrix:
        """Create sparse interaction matrix."""
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
    
    def get_user_candidates(self, user_id: str, n_candidates: int = 100) -> List[Dict[str, Any]]:
        """Get top-N candidate items for a user."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if user_id not in self.user_mapping:
            logger.warning("User not in training set", user_id=user_id, domain=self.domain)
            return []
        
        # Get user index
        user_idx = self.user_mapping[user_id]
        
        # Get recommendations
        item_scores = self.model.recommend(
            user_idx, 
            self.model.item_factors, 
            N=n_candidates,
            filter_already_liked_items=True
        )
        
        # Convert to list of candidates
        candidates = []
        for item_idx, score in item_scores:
            item_id = self.reverse_item_mapping[item_idx]
            candidates.append({
                'item_id': item_id,
                'score': float(score),
                'source': 'als'
            })
        
        return candidates
    
    def get_similar_items(self, item_id: str, n_similar: int = 20) -> List[Dict[str, Any]]:
        """Get similar items for a given item."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        if item_id not in self.item_mapping:
            logger.warning("Item not in training set", item_id=item_id, domain=self.domain)
            return []
        
        # Get item index
        item_idx = self.item_mapping[item_id]
        
        # Get similar items
        similar_items = self.model.similar_items(item_idx, n_similar + 1)  # +1 to exclude self
        
        # Convert to list
        candidates = []
        for similar_idx, score in similar_items[1:]:  # Skip first (self)
            similar_item_id = self.reverse_item_mapping[similar_idx]
            candidates.append({
                'item_id': similar_item_id,
                'score': float(score),
                'source': 'als_similar'
            })
        
        return candidates
    
    def get_user_factors(self, user_id: str) -> Optional[np.ndarray]:
        """Get user factors for a specific user."""
        if self.model is None or user_id not in self.user_mapping:
            return None
        
        user_idx = self.user_mapping[user_id]
        return self.user_factors[user_idx]
    
    def get_item_factors(self, item_id: str) -> Optional[np.ndarray]:
        """Get item factors for a specific item."""
        if self.model is None or item_id not in self.item_mapping:
            return None
        
        item_idx = self.item_mapping[item_id]
        return self.item_factors[item_idx]
    
    def predict_score(self, user_id: str, item_id: str) -> float:
        """Predict interaction score for user-item pair."""
        if self.model is None:
            return 0.0
        
        user_factors = self.get_user_factors(user_id)
        item_factors = self.get_item_factors(item_id)
        
        if user_factors is None or item_factors is None:
            return 0.0
        
        return float(np.dot(user_factors, item_factors))
    
    def save(self, domain: str):
        """Save ALS model and mappings."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model:
            with open(artifacts_dir / "als_model.pkl", "wb") as f:
                pickle.dump(self.model, f)
        
        # Save mappings
        mappings = {
            'user_mapping': self.user_mapping,
            'item_mapping': self.item_mapping,
            'reverse_user_mapping': self.reverse_user_mapping,
            'reverse_item_mapping': self.reverse_item_mapping
        }
        
        with open(artifacts_dir / "als_mappings.pkl", "wb") as f:
            pickle.dump(mappings, f)
        
        # Save factors
        if self.user_factors is not None and self.item_factors is not None:
            np.savez_compressed(
                artifacts_dir / "als_factors.npz",
                user_factors=self.user_factors,
                item_factors=self.item_factors
            )
        
        logger.info("ALS model saved", domain=domain)
    
    def load(self, domain: str):
        """Load ALS model and mappings."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        
        # Load model
        model_path = artifacts_dir / "als_model.pkl"
        if model_path.exists():
            with open(model_path, "rb") as f:
                self.model = pickle.load(f)
        
        # Load mappings
        mappings_path = artifacts_dir / "als_mappings.pkl"
        if mappings_path.exists():
            with open(mappings_path, "rb") as f:
                mappings = pickle.load(f)
                self.user_mapping = mappings['user_mapping']
                self.item_mapping = mappings['item_mapping']
                self.reverse_user_mapping = mappings['reverse_user_mapping']
                self.reverse_item_mapping = mappings['reverse_item_mapping']
        
        # Load factors
        factors_path = artifacts_dir / "als_factors.npz"
        if factors_path.exists():
            factors = np.load(factors_path)
            self.user_factors = factors['user_factors']
            self.item_factors = factors['item_factors']
        
        logger.info("ALS model loaded", domain=domain) 