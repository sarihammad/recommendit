"""LightGBM ranking model for recommendation scoring."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import structlog
import lightgbm as lgb
from sklearn.preprocessing import StandardScaler, LabelEncoder
import pickle
from pathlib import Path
from ..service.config import config

logger = structlog.get_logger(__name__)

class LightGBMRanker:
    """LightGBM-based ranking model for recommendations."""
    
    def __init__(self, domain: str):
        """Initialize LightGBM ranker."""
        self.domain = domain
        self.model = None
        self.feature_names = []
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        
    def fit(self, train_data: pd.DataFrame, val_data: pd.DataFrame = None):
        """Fit LightGBM ranking model."""
        logger.info("Fitting LightGBM ranker", domain=self.domain)
        
        # Prepare training data
        X_train, y_train, groups_train = self._prepare_training_data(train_data)
        
        # Prepare validation data if provided
        X_val, y_val, groups_val = None, None, None
        if val_data is not None:
            X_val, y_val, groups_val = self._prepare_training_data(val_data)
        
        # Create LightGBM datasets
        train_dataset = lgb.Dataset(
            X_train, 
            label=y_train, 
            group=groups_train,
            feature_name=self.feature_names
        )
        
        val_dataset = None
        if X_val is not None:
            val_dataset = lgb.Dataset(
                X_val, 
                label=y_val, 
                group=groups_val,
                feature_name=self.feature_names,
                reference=train_dataset
            )
        
        # Set up parameters
        params = {
            'objective': config.LGBM_OBJECTIVE,
            'metric': config.LGBM_METRIC,
            'ndcg_eval_at': config.LGBM_NDCG_EVAL_AT,
            'num_leaves': config.LGBM_NUM_LEAVES,
            'learning_rate': config.LGBM_LEARNING_RATE,
            'num_iterations': config.LGBM_NUM_ITERATIONS,
            'random_state': 42,
            'verbose': -1
        }
        
        # Train model
        self.model = lgb.train(
            params,
            train_dataset,
            valid_sets=[val_dataset] if val_dataset else None,
            callbacks=[lgb.log_evaluation(period=10)]
        )
        
        logger.info("LightGBM ranker fitted", 
                   domain=self.domain,
                   features=len(self.feature_names))
    
    def _prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[int]]:
        """Prepare training data for LightGBM."""
        # Extract features and labels
        feature_cols = [col for col in data.columns if col.startswith('feature_')]
        self.feature_names = feature_cols
        
        X = data[feature_cols].values
        y = data['label'].values if 'label' in data.columns else np.zeros(len(data))
        
        # Create groups (number of items per user)
        if 'user_id' in data.columns:
            groups = data.groupby('user_id').size().tolist()
        else:
            groups = [len(data)]  # Single group
        
        return X, y, groups
    
    def predict(self, features_df: pd.DataFrame) -> np.ndarray:
        """Predict ranking scores."""
        if self.model is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        # Extract features
        feature_cols = [col for col in features_df.columns if col.startswith('feature_')]
        X = features_df[feature_cols].values
        
        # Ensure all expected features are present
        missing_features = set(self.feature_names) - set(feature_cols)
        if missing_features:
            logger.warning("Missing features", missing_features=missing_features)
            # Add missing features with zeros
            for feature in missing_features:
                features_df[f'feature_{feature}'] = 0
            X = features_df[self.feature_names].values
        
        # Predict
        scores = self.model.predict(X)
        return scores
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance scores."""
        if self.model is None:
            return {}
        
        importance = self.model.feature_importance(importance_type='gain')
        return dict(zip(self.feature_names, importance))
    
    def save(self, domain: str):
        """Save LightGBM model and feature processors."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model
        if self.model:
            model_path = artifacts_dir / "lgbm_model.txt"
            self.model.save_model(str(model_path))
        
        # Save feature information
        feature_info = {
            'feature_names': self.feature_names,
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_stats': self.feature_stats
        }
        
        with open(artifacts_dir / "lgbm_features.pkl", "wb") as f:
            pickle.dump(feature_info, f)
        
        logger.info("LightGBM ranker saved", domain=domain)
    
    def load(self, domain: str):
        """Load LightGBM model and feature processors."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        
        # Load model
        model_path = artifacts_dir / "lgbm_model.txt"
        if model_path.exists():
            self.model = lgb.Booster(model_file=str(model_path))
        
        # Load feature information
        feature_path = artifacts_dir / "lgbm_features.pkl"
        if feature_path.exists():
            with open(feature_path, "rb") as f:
                feature_info = pickle.load(f)
                self.feature_names = feature_info['feature_names']
                self.scalers = feature_info['scalers']
                self.encoders = feature_info['encoders']
                self.feature_stats = feature_info['feature_stats']
        
        logger.info("LightGBM ranker loaded", domain=domain)

class FeatureEngineer:
    """Feature engineering for ranking model."""
    
    def __init__(self, domain: str):
        """Initialize feature engineer."""
        self.domain = domain
        self.scalers = {}
        self.encoders = {}
        self.feature_stats = {}
        
    def create_ranking_features(self, 
                              user_id: str,
                              candidates: List[Dict[str, Any]],
                              als_model,
                              item_item_model,
                              content_model,
                              user_features: Dict[str, Any],
                              items_df: pd.DataFrame) -> pd.DataFrame:
        """Create features for ranking candidates."""
        logger.info("Creating ranking features", user_id=user_id, num_candidates=len(candidates))
        
        features_list = []
        
        for candidate in candidates:
            item_id = candidate['item_id']
            source = candidate['source']
            
            # Get item metadata
            item_meta = items_df[items_df['item_id'] == item_id].iloc[0] if len(items_df[items_df['item_id'] == item_id]) > 0 else None
            
            # Initialize feature dict
            features = {
                'user_id': user_id,
                'item_id': item_id,
                'source': source
            }
            
            # Collaborative filtering features
            features.update(self._create_cf_features(user_id, item_id, als_model, item_item_model))
            
            # Content-based features
            features.update(self._create_content_features(user_id, item_id, content_model, item_meta))
            
            # User features
            features.update(self._create_user_features(user_id, user_features))
            
            # Item features
            features.update(self._create_item_features(item_id, item_meta))
            
            # Context features
            features.update(self._create_context_features(user_id, item_id, source))
            
            features_list.append(features)
        
        # Convert to DataFrame
        features_df = pd.DataFrame(features_list)
        
        # Normalize and encode features
        features_df = self._normalize_features(features_df)
        
        return features_df
    
    def _create_cf_features(self, user_id: str, item_id: str, als_model, item_item_model) -> Dict[str, float]:
        """Create collaborative filtering features."""
        features = {}
        
        # ALS features
        if als_model:
            # ALS prediction score
            features['feature_als_score'] = als_model.predict_score(user_id, item_id)
            
            # ALS user factors (if available)
            user_factors = als_model.get_user_factors(user_id)
            if user_factors is not None:
                for i, factor in enumerate(user_factors[:10]):  # First 10 factors
                    features[f'feature_als_user_factor_{i}'] = float(factor)
            
            # ALS item factors (if available)
            item_factors = als_model.get_item_factors(item_id)
            if item_factors is not None:
                for i, factor in enumerate(item_factors[:10]):  # First 10 factors
                    features[f'feature_als_item_factor_{i}'] = float(factor)
        
        # Item-item features
        if item_item_model:
            # Item-item similarity score
            features['feature_item_item_score'] = item_item_model.predict_score(user_id, item_id)
        
        return features
    
    def _create_content_features(self, user_id: str, item_id: str, content_model, item_meta) -> Dict[str, float]:
        """Create content-based features."""
        features = {}
        
        if content_model and item_meta is not None:
            # Content embedding score (if user history available)
            # This would require user history - simplified for now
            features['feature_content_score'] = 0.0
            
            # Genre features
            if 'genres' in item_meta:
                genres = item_meta['genres']
                if isinstance(genres, list):
                    features['feature_genre_count'] = len(genres)
                    features['feature_has_fiction'] = 1.0 if 'Fiction' in genres else 0.0
                    features['feature_has_nonfiction'] = 1.0 if 'Non-Fiction' in genres else 0.0
                else:
                    features['feature_genre_count'] = 0
                    features['feature_has_fiction'] = 0.0
                    features['feature_has_nonfiction'] = 0.0
            
            # Year features
            if 'year' in item_meta and item_meta['year'] is not None:
                features['feature_year'] = float(item_meta['year'])
                features['feature_is_recent'] = 1.0 if item_meta['year'] >= 2000 else 0.0
            else:
                features['feature_year'] = 0.0
                features['feature_is_recent'] = 0.0
        
        return features
    
    def _create_user_features(self, user_id: str, user_features: Dict[str, Any]) -> Dict[str, float]:
        """Create user-based features."""
        features = {}
        
        # User interaction count
        features['feature_user_interaction_count'] = user_features.get('interaction_count', 0)
        
        # User average rating
        features['feature_user_avg_rating'] = user_features.get('avg_rating', 0.0)
        
        # User genre preferences (top genres)
        for key, value in user_features.items():
            if key.startswith('genre_pref_'):
                features[f'feature_{key}'] = value
        
        return features
    
    def _create_item_features(self, item_id: str, item_meta) -> Dict[str, float]:
        """Create item-based features."""
        features = {}
        
        if item_meta is not None:
            # Title length
            title = item_meta.get('title', '')
            features['feature_title_length'] = len(title)
            
            # Description length
            description = item_meta.get('description', '')
            features['feature_description_length'] = len(description)
            
            # Author popularity (if available)
            author = item_meta.get('author', '')
            features['feature_has_author'] = 1.0 if author and author != 'Unknown' else 0.0
        
        return features
    
    def _create_context_features(self, user_id: str, item_id: str, source: str) -> Dict[str, float]:
        """Create context features."""
        features = {}
        
        # Source features
        features['feature_source_als'] = 1.0 if source == 'als' else 0.0
        features['feature_source_item_item'] = 1.0 if source == 'item_item' else 0.0
        features['feature_source_content'] = 1.0 if 'content' in source else 0.0
        
        return features
    
    def _normalize_features(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Normalize and encode features."""
        # Get feature columns
        feature_cols = [col for col in features_df.columns if col.startswith('feature_')]
        
        # Normalize numerical features
        for col in feature_cols:
            if col not in self.scalers:
                self.scalers[col] = StandardScaler()
                features_df[col] = self.scalers[col].fit_transform(features_df[col].values.reshape(-1, 1))
            else:
                features_df[col] = self.scalers[col].transform(features_df[col].values.reshape(-1, 1))
        
        # Store feature statistics
        for col in feature_cols:
            self.feature_stats[col] = {
                'mean': features_df[col].mean(),
                'std': features_df[col].std(),
                'min': features_df[col].min(),
                'max': features_df[col].max()
            }
        
        return features_df
    
    def save(self, domain: str):
        """Save feature engineering artifacts."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        artifacts_dir.mkdir(parents=True, exist_ok=True)
        
        # Save scalers and encoders
        feature_processors = {
            'scalers': self.scalers,
            'encoders': self.encoders,
            'feature_stats': self.feature_stats
        }
        
        with open(artifacts_dir / "feature_processors.pkl", "wb") as f:
            pickle.dump(feature_processors, f)
        
        logger.info("Feature engineering artifacts saved", domain=domain)
    
    def load(self, domain: str):
        """Load feature engineering artifacts."""
        artifacts_dir = Path(config.ARTIFACTS_DIR) / domain
        processors_path = artifacts_dir / "feature_processors.pkl"
        
        if processors_path.exists():
            with open(processors_path, "rb") as f:
                feature_processors = pickle.load(f)
                self.scalers = feature_processors['scalers']
                self.encoders = feature_processors['encoders']
                self.feature_stats = feature_processors['feature_stats']
            
            logger.info("Feature engineering artifacts loaded", domain=domain) 