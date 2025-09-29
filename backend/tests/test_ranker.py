"""Tests for LightGBM ranking model and feature engineering."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from recsys.models.ranker_lgbm import LightGBMRanker, FeatureEngineer
from recsys.service.config import config

class TestLightGBMRanker:
    """Test LightGBM ranking model."""
    
    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for testing."""
        # Create sample features
        n_samples = 100
        feature_data = []
        
        for i in range(n_samples):
            features = {
                'user_id': f'user_{i % 10}',
                'item_id': f'item_{i % 20}',
                'cf_score': np.random.random(),
                'content_score': np.random.random(),
                'user_interaction_count': np.random.randint(1, 100),
                'item_popularity': np.random.random(),
                'genre_overlap': np.random.random(),
                'author_match': np.random.randint(0, 2),
                'year_diff': np.random.randint(0, 50),
                'label': np.random.randint(0, 2)  # Binary labels
            }
            feature_data.append(features)
        
        return pd.DataFrame(feature_data)
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_ranker_initialization(self):
        """Test LightGBM ranker initialization."""
        ranker = LightGBMRanker('test')
        
        assert ranker.domain == 'test'
        assert ranker.model is None
        assert isinstance(ranker.feature_names, list)
        assert isinstance(ranker.scalers, dict)
        assert isinstance(ranker.encoders, dict)
        assert isinstance(ranker.feature_stats, dict)
    
    def test_ranker_fit(self, sample_training_data):
        """Test LightGBM ranker fitting."""
        ranker = LightGBMRanker('test')
        
        # Prepare training data
        feature_cols = ['cf_score', 'content_score', 'user_interaction_count', 
                       'item_popularity', 'genre_overlap', 'author_match', 'year_diff']
        X = sample_training_data[feature_cols]
        y = sample_training_data['label']
        
        # Fit the model
        ranker.fit(sample_training_data)
        
        # Check that model was trained
        assert ranker.model is not None
        assert len(ranker.feature_names) > 0
        
        # Check that feature names are set
        for col in feature_cols:
            assert col in ranker.feature_names
    
    def test_ranker_predict(self, sample_training_data):
        """Test LightGBM ranker prediction."""
        ranker = LightGBMRanker('test')
        ranker.fit(sample_training_data)
        
        # Create test data
        test_data = sample_training_data.head(10)
        predictions = ranker.predict(test_data)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(test_data)
        assert all(isinstance(pred, (int, float)) for pred in predictions)
    
    def test_ranker_save_and_load(self, sample_training_data, temp_artifacts_dir):
        """Test LightGBM ranker saving and loading."""
        ranker = LightGBMRanker('test')
        ranker.fit(sample_training_data)
        
        # Temporarily change artifacts directory
        original_artifacts_dir = config.ARTIFACTS_DIR
        config.ARTIFACTS_DIR = temp_artifacts_dir
        
        try:
            # Save model
            ranker.save('test')
            
            # Create new model and load
            loaded_ranker = LightGBMRanker('test')
            loaded_ranker.load('test')
            
            # Check that loaded model has same properties
            assert loaded_ranker.domain == ranker.domain
            assert loaded_ranker.model is not None
            assert len(loaded_ranker.feature_names) == len(ranker.feature_names)
            
            # Check that predictions are similar
            test_data = sample_training_data.head(5)
            original_predictions = ranker.predict(test_data)
            loaded_predictions = loaded_ranker.predict(test_data)
            
            # Predictions should be the same
            np.testing.assert_array_almost_equal(original_predictions, loaded_predictions)
            
        finally:
            # Restore original artifacts directory
            config.ARTIFACTS_DIR = original_artifacts_dir

class TestFeatureEngineer:
    """Test feature engineering functionality."""
    
    @pytest.fixture
    def sample_candidates(self):
        """Create sample candidates for testing."""
        return [
            {
                'item_id': 'item_1',
                'score': 0.8,
                'source': 'als'
            },
            {
                'item_id': 'item_2',
                'score': 0.6,
                'source': 'content'
            },
            {
                'item_id': 'item_3',
                'score': 0.7,
                'source': 'item_item'
            }
        ]
    
    @pytest.fixture
    def sample_user_features(self):
        """Create sample user features for testing."""
        return {
            'interaction_count': 25,
            'avg_rating': 4.2,
            'preferred_genres': ['Action', 'Adventure'],
            'preferred_authors': ['Author A', 'Author B']
        }
    
    @pytest.fixture
    def sample_items_df(self):
        """Create sample items dataframe for testing."""
        return pd.DataFrame({
            'item_id': ['item_1', 'item_2', 'item_3'],
            'title': ['Book 1', 'Book 2', 'Book 3'],
            'author': ['Author A', 'Author B', 'Author C'],
            'genres': ['Action|Adventure', 'Romance|Drama', 'Sci-Fi|Action'],
            'year': [2020, 2019, 2021],
            'popularity': [100, 50, 75]
        })
    
    @pytest.fixture
    def mock_models(self):
        """Create mock models for testing."""
        class MockALSModel:
            def predict_score(self, user_id, item_id):
                return 0.8 if item_id == 'item_1' else 0.5
        
        class MockItemItemModel:
            def predict_score(self, user_id, item_id):
                return 0.7 if item_id == 'item_2' else 0.4
        
        class MockContentModel:
            def predict_score(self, user_history, item_id):
                return 0.6 if item_id == 'item_3' else 0.3
        
        return {
            'als_model': MockALSModel(),
            'item_item_model': MockItemItemModel(),
            'content_model': MockContentModel()
        }
    
    def test_feature_engineer_initialization(self):
        """Test feature engineer initialization."""
        feature_engineer = FeatureEngineer('test')
        
        assert feature_engineer.domain == 'test'
        assert isinstance(feature_engineer.scalers, dict)
        assert isinstance(feature_engineer.encoders, dict)
        assert isinstance(feature_engineer.feature_stats, dict)
    
    def test_create_ranking_features(self, sample_candidates, sample_user_features, 
                                   sample_items_df, mock_models):
        """Test creating ranking features."""
        feature_engineer = FeatureEngineer('test')
        
        features_df = feature_engineer.create_ranking_features(
            user_id='user_1',
            candidates=sample_candidates,
            als_model=mock_models['als_model'],
            item_item_model=mock_models['item_item_model'],
            content_model=mock_models['content_model'],
            user_features=sample_user_features,
            items_df=sample_items_df
        )
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(sample_candidates)
        
        # Check that expected features are present
        expected_features = [
            'cf_score', 'content_score', 'user_interaction_count',
            'item_popularity', 'genre_overlap', 'author_match', 'year_diff'
        ]
        
        for feature in expected_features:
            assert feature in features_df.columns
        
        # Check that features are numeric
        for feature in expected_features:
            if feature in features_df.columns:
                assert pd.api.types.is_numeric_dtype(features_df[feature])
    
    def test_create_cf_features(self, sample_candidates, mock_models):
        """Test creating collaborative filtering features."""
        feature_engineer = FeatureEngineer('test')
        
        cf_features = feature_engineer._create_cf_features(
            user_id='user_1',
            candidates=sample_candidates,
            als_model=mock_models['als_model'],
            item_item_model=mock_models['item_item_model']
        )
        
        assert isinstance(cf_features, pd.DataFrame)
        assert len(cf_features) == len(sample_candidates)
        assert 'cf_score' in cf_features.columns
    
    def test_create_content_features(self, sample_candidates, sample_user_features, 
                                   sample_items_df, mock_models):
        """Test creating content-based features."""
        feature_engineer = FeatureEngineer('test')
        
        content_features = feature_engineer._create_content_features(
            candidates=sample_candidates,
            content_model=mock_models['content_model'],
            items_df=sample_items_df
        )
        
        assert isinstance(content_features, pd.DataFrame)
        assert len(content_features) == len(sample_candidates)
        assert 'content_score' in content_features.columns
    
    def test_create_user_features(self, sample_candidates, sample_user_features):
        """Test creating user features."""
        feature_engineer = FeatureEngineer('test')
        
        user_features = feature_engineer._create_user_features(
            candidates=sample_candidates,
            user_features=sample_user_features
        )
        
        assert isinstance(user_features, pd.DataFrame)
        assert len(user_features) == len(sample_candidates)
        assert 'user_interaction_count' in user_features.columns
    
    def test_create_item_features(self, sample_candidates, sample_items_df):
        """Test creating item features."""
        feature_engineer = FeatureEngineer('test')
        
        item_features = feature_engineer._create_item_features(
            candidates=sample_candidates,
            items_df=sample_items_df
        )
        
        assert isinstance(item_features, pd.DataFrame)
        assert len(item_features) == len(sample_candidates)
        assert 'item_popularity' in item_features.columns
    
    def test_create_context_features(self, sample_candidates, sample_user_features, 
                                   sample_items_df):
        """Test creating context features."""
        feature_engineer = FeatureEngineer('test')
        
        context_features = feature_engineer._create_context_features(
            candidates=sample_candidates,
            user_features=sample_user_features,
            items_df=sample_items_df
        )
        
        assert isinstance(context_features, pd.DataFrame)
        assert len(context_features) == len(sample_candidates)
        
        # Check for context-specific features
        expected_context_features = ['genre_overlap', 'author_match', 'year_diff']
        for feature in expected_context_features:
            if feature in context_features.columns:
                assert pd.api.types.is_numeric_dtype(context_features[feature])
    
    def test_feature_engineer_save_and_load(self, temp_artifacts_dir):
        """Test feature engineer saving and loading."""
        feature_engineer = FeatureEngineer('test')
        
        # Add some test data
        feature_engineer.scalers['test_scaler'] = 'scaler_data'
        feature_engineer.encoders['test_encoder'] = 'encoder_data'
        feature_engineer.feature_stats['test_stat'] = 'stat_data'
        
        # Temporarily change artifacts directory
        original_artifacts_dir = config.ARTIFACTS_DIR
        config.ARTIFACTS_DIR = temp_artifacts_dir
        
        try:
            # Save feature engineer
            feature_engineer.save('test')
            
            # Create new feature engineer and load
            loaded_engineer = FeatureEngineer('test')
            loaded_engineer.load('test')
            
            # Check that loaded engineer has same properties
            assert loaded_engineer.domain == feature_engineer.domain
            assert 'test_scaler' in loaded_engineer.scalers
            assert 'test_encoder' in loaded_engineer.encoders
            assert 'test_stat' in loaded_engineer.feature_stats
            
        finally:
            # Restore original artifacts directory
            config.ARTIFACTS_DIR = original_artifacts_dir

class TestRankingIntegration:
    """Integration tests for ranking system."""
    
    def test_end_to_end_ranking(self):
        """Test end-to-end ranking pipeline."""
        # Create sample data
        candidates = [
            {'item_id': 'item_1', 'score': 0.8, 'source': 'als'},
            {'item_id': 'item_2', 'score': 0.6, 'source': 'content'},
            {'item_id': 'item_3', 'score': 0.7, 'source': 'item_item'}
        ]
        
        user_features = {
            'interaction_count': 25,
            'avg_rating': 4.2,
            'preferred_genres': ['Action', 'Adventure']
        }
        
        items_df = pd.DataFrame({
            'item_id': ['item_1', 'item_2', 'item_3'],
            'title': ['Book 1', 'Book 2', 'Book 3'],
            'author': ['Author A', 'Author B', 'Author C'],
            'genres': ['Action|Adventure', 'Romance|Drama', 'Sci-Fi|Action'],
            'year': [2020, 2019, 2021],
            'popularity': [100, 50, 75]
        })
        
        # Create mock models
        class MockModel:
            def predict_score(self, *args):
                return 0.5
        
        mock_models = {
            'als_model': MockModel(),
            'item_item_model': MockModel(),
            'content_model': MockModel()
        }
        
        # Create features
        feature_engineer = FeatureEngineer('test')
        features_df = feature_engineer.create_ranking_features(
            user_id='user_1',
            candidates=candidates,
            als_model=mock_models['als_model'],
            item_item_model=mock_models['item_item_model'],
            content_model=mock_models['content_model'],
            user_features=user_features,
            items_df=items_df
        )
        
        # Train ranker
        ranker = LightGBMRanker('test')
        
        # Add labels for training
        features_df['label'] = [1, 0, 1]  # Binary labels
        
        ranker.fit(features_df)
        
        # Make predictions
        predictions = ranker.predict(features_df)
        
        assert isinstance(predictions, np.ndarray)
        assert len(predictions) == len(candidates)
    
    def test_ranking_with_missing_data(self):
        """Test ranking with missing data."""
        feature_engineer = FeatureEngineer('test')
        
        # Create candidates with missing item data
        candidates = [
            {'item_id': 'item_1', 'score': 0.8, 'source': 'als'},
            {'item_id': 'missing_item', 'score': 0.6, 'source': 'content'}
        ]
        
        items_df = pd.DataFrame({
            'item_id': ['item_1'],
            'title': ['Book 1'],
            'author': ['Author A'],
            'genres': ['Action|Adventure'],
            'year': [2020],
            'popularity': [100]
        })
        
        class MockModel:
            def predict_score(self, *args):
                return 0.5
        
        mock_models = {
            'als_model': MockModel(),
            'item_item_model': MockModel(),
            'content_model': MockModel()
        }
        
        user_features = {'interaction_count': 25}
        
        # Should handle missing data gracefully
        features_df = feature_engineer.create_ranking_features(
            user_id='user_1',
            candidates=candidates,
            als_model=mock_models['als_model'],
            item_item_model=mock_models['item_item_model'],
            content_model=mock_models['content_model'],
            user_features=user_features,
            items_df=items_df
        )
        
        assert isinstance(features_df, pd.DataFrame)
        assert len(features_df) == len(candidates)

if __name__ == "__main__":
    pytest.main([__file__]) 