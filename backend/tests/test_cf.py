"""Tests for collaborative filtering models."""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil

from recsys.models.cf_als import ALSRecommender
from recsys.models.cf_item_item import ItemItemRecommender
from recsys.service.config import config

class TestALSRecommender:
    """Test ALS collaborative filtering model."""
    
    @pytest.fixture
    def sample_interactions(self):
        """Create sample interaction data for testing."""
        return pd.DataFrame({
            'user_id': ['1', '2', '3', '1', '2', '4', '3', '5'] * 5,
            'item_id': ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C'] * 5,
            'weight': [5.0, 4.0, 3.0, 5.0, 4.0, 3.0, 5.0, 4.0] * 5,
            'timestamp': pd.date_range('2020-01-01', periods=40)
        })
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_als_initialization(self):
        """Test ALS model initialization."""
        model = ALSRecommender('test')
        
        assert model.domain == 'test'
        assert model.model is None
        assert isinstance(model.user_mapping, dict)
        assert isinstance(model.item_mapping, dict)
    
    def test_als_fit(self, sample_interactions):
        """Test ALS model fitting."""
        model = ALSRecommender('test')
        
        # Fit the model
        model.fit(sample_interactions)
        
        # Check that model was trained
        assert model.model is not None
        assert model.user_factors is not None
        assert model.item_factors is not None
        
        # Check mappings
        assert len(model.user_mapping) > 0
        assert len(model.item_mapping) > 0
        assert len(model.reverse_user_mapping) > 0
        assert len(model.reverse_item_mapping) > 0
        
        # Check factors shape
        assert model.user_factors.shape[0] == len(model.user_mapping)
        assert model.item_factors.shape[0] == len(model.item_mapping)
        assert model.user_factors.shape[1] == model.item_factors.shape[1]
    
    def test_als_get_user_candidates(self, sample_interactions):
        """Test getting user candidates from ALS model."""
        model = ALSRecommender('test')
        model.fit(sample_interactions)
        
        # Get candidates for a user
        user_id = '1'
        candidates = model.get_user_candidates(user_id, n_candidates=10)
        
        assert isinstance(candidates, list)
        assert len(candidates) <= 10
        
        if candidates:
            # Check candidate structure
            candidate = candidates[0]
            assert 'item_id' in candidate
            assert 'score' in candidate
            assert 'source' in candidate
            assert candidate['source'] == 'als'
            
            # Check that scores are numeric
            assert isinstance(candidate['score'], (int, float))
    
    def test_als_get_similar_items(self, sample_interactions):
        """Test getting similar items from ALS model."""
        model = ALSRecommender('test')
        model.fit(sample_interactions)
        
        # Get similar items
        item_id = 'A'
        similar_items = model.get_similar_items(item_id, n_similar=5)
        
        assert isinstance(similar_items, list)
        assert len(similar_items) <= 5
        
        if similar_items:
            # Check similar item structure
            similar_item = similar_items[0]
            assert 'item_id' in similar_item
            assert 'score' in similar_item
            assert 'source' in similar_item
            assert similar_item['source'] == 'als'
            
            # Check that scores are numeric
            assert isinstance(similar_item['score'], (int, float))
    
    def test_als_predict_score(self, sample_interactions):
        """Test ALS score prediction."""
        model = ALSRecommender('test')
        model.fit(sample_interactions)
        
        # Predict score for a user-item pair
        user_id = '1'
        item_id = 'A'
        score = model.predict_score(user_id, item_id)
        
        assert isinstance(score, (int, float))
        assert score >= 0  # Scores should be non-negative
    
    def test_als_save_and_load(self, sample_interactions, temp_artifacts_dir):
        """Test ALS model saving and loading."""
        model = ALSRecommender('test')
        model.fit(sample_interactions)
        
        # Temporarily change artifacts directory
        original_artifacts_dir = config.ARTIFACTS_DIR
        config.ARTIFACTS_DIR = temp_artifacts_dir
        
        try:
            # Save model
            model.save('test')
            
            # Create new model and load
            loaded_model = ALSRecommender('test')
            loaded_model.load('test')
            
            # Check that loaded model has same properties
            assert loaded_model.domain == model.domain
            assert loaded_model.model is not None
            assert loaded_model.user_factors is not None
            assert loaded_model.item_factors is not None
            
            # Check mappings
            assert len(loaded_model.user_mapping) == len(model.user_mapping)
            assert len(loaded_model.item_mapping) == len(model.item_mapping)
            
            # Check that predictions are similar
            user_id = '1'
            item_id = 'A'
            original_score = model.predict_score(user_id, item_id)
            loaded_score = loaded_model.predict_score(user_id, item_id)
            
            # Scores should be close (allowing for small numerical differences)
            assert abs(original_score - loaded_score) < 1e-6
            
        finally:
            # Restore original artifacts directory
            config.ARTIFACTS_DIR = original_artifacts_dir

class TestItemItemRecommender:
    """Test Item-Item collaborative filtering model."""
    
    @pytest.fixture
    def sample_interactions(self):
        """Create sample interaction data for testing."""
        return pd.DataFrame({
            'user_id': ['1', '2', '3', '1', '2', '4', '3', '5'] * 5,
            'item_id': ['A', 'B', 'C', 'D', 'E', 'A', 'B', 'C'] * 5,
            'weight': [5.0, 4.0, 3.0, 5.0, 4.0, 3.0, 5.0, 4.0] * 5,
            'timestamp': pd.date_range('2020-01-01', periods=40)
        })
    
    @pytest.fixture
    def temp_artifacts_dir(self):
        """Create temporary artifacts directory for testing."""
        temp_dir = tempfile.mkdtemp()
        yield temp_dir
        shutil.rmtree(temp_dir)
    
    def test_item_item_initialization(self):
        """Test Item-Item model initialization."""
        model = ItemItemRecommender('test')
        
        assert model.domain == 'test'
        assert model.item_similarity_matrix is None
        assert isinstance(model.item_mapping, dict)
        assert isinstance(model.user_mapping, dict)
    
    def test_item_item_fit(self, sample_interactions):
        """Test Item-Item model fitting."""
        model = ItemItemRecommender('test')
        
        # Fit the model
        model.fit(sample_interactions)
        
        # Check that similarity matrix was created
        assert model.item_similarity_matrix is not None
        assert model.user_item_matrix is not None
        
        # Check mappings
        assert len(model.item_mapping) > 0
        assert len(model.user_mapping) > 0
        assert len(model.reverse_item_mapping) > 0
        assert len(model.reverse_user_mapping) > 0
        
        # Check similarity matrix shape
        n_items = len(model.item_mapping)
        assert model.item_similarity_matrix.shape == (n_items, n_items)
        
        # Check that diagonal is 1.0 (self-similarity)
        np.testing.assert_array_almost_equal(
            np.diag(model.item_similarity_matrix), 
            np.ones(n_items)
        )
    
    def test_item_item_get_similar_items(self, sample_interactions):
        """Test getting similar items from Item-Item model."""
        model = ItemItemRecommender('test')
        model.fit(sample_interactions)
        
        # Get similar items
        item_id = 'A'
        similar_items = model.get_similar_items(item_id, n_similar=5)
        
        assert isinstance(similar_items, list)
        assert len(similar_items) <= 5
        
        if similar_items:
            # Check similar item structure
            similar_item = similar_items[0]
            assert 'item_id' in similar_item
            assert 'score' in similar_item
            assert 'source' in similar_item
            assert similar_item['source'] == 'item_item'
            
            # Check that scores are numeric and between 0 and 1
            assert isinstance(similar_item['score'], (int, float))
            assert 0 <= similar_item['score'] <= 1
    
    def test_item_item_get_user_candidates(self, sample_interactions):
        """Test getting user candidates from Item-Item model."""
        model = ItemItemRecommender('test')
        model.fit(sample_interactions)
        
        # Get candidates for a user
        user_id = '1'
        candidates = model.get_user_candidates(user_id, n_candidates=10)
        
        assert isinstance(candidates, list)
        assert len(candidates) <= 10
        
        if candidates:
            # Check candidate structure
            candidate = candidates[0]
            assert 'item_id' in candidate
            assert 'score' in candidate
            assert 'source' in candidate
            assert candidate['source'] == 'item_item'
            
            # Check that scores are numeric
            assert isinstance(candidate['score'], (int, float))
    
    def test_item_item_get_user_candidates_from_history(self, sample_interactions):
        """Test getting candidates from user history."""
        model = ItemItemRecommender('test')
        model.fit(sample_interactions)
        
        # Get candidates from user history
        user_id = '1'
        user_history = ['A', 'B']  # User has interacted with items A and B
        candidates = model.get_user_candidates_from_history(user_id, user_history, n_candidates=10)
        
        assert isinstance(candidates, list)
        assert len(candidates) <= 10
        
        if candidates:
            # Check candidate structure
            candidate = candidates[0]
            assert 'item_id' in candidate
            assert 'score' in candidate
            assert 'source' in candidate
            assert candidate['source'] == 'item_item'
    
    def test_item_item_predict_score(self, sample_interactions):
        """Test Item-Item score prediction."""
        model = ItemItemRecommender('test')
        model.fit(sample_interactions)
        
        # Predict score for a user-item pair
        user_id = '1'
        item_id = 'A'
        score = model.predict_score(user_id, item_id)
        
        assert isinstance(score, (int, float))
        assert score >= 0  # Scores should be non-negative
    
    def test_item_item_save_and_load(self, sample_interactions, temp_artifacts_dir):
        """Test Item-Item model saving and loading."""
        model = ItemItemRecommender('test')
        model.fit(sample_interactions)
        
        # Temporarily change artifacts directory
        original_artifacts_dir = config.ARTIFACTS_DIR
        config.ARTIFACTS_DIR = temp_artifacts_dir
        
        try:
            # Save model
            model.save('test')
            
            # Create new model and load
            loaded_model = ItemItemRecommender('test')
            loaded_model.load('test')
            
            # Check that loaded model has same properties
            assert loaded_model.domain == model.domain
            assert loaded_model.item_similarity_matrix is not None
            assert loaded_model.user_item_matrix is not None
            
            # Check mappings
            assert len(loaded_model.item_mapping) == len(model.item_mapping)
            assert len(loaded_model.user_mapping) == len(model.user_mapping)
            
            # Check that similarity matrix is the same
            np.testing.assert_array_almost_equal(
                loaded_model.item_similarity_matrix,
                model.item_similarity_matrix
            )
            
        finally:
            # Restore original artifacts directory
            config.ARTIFACTS_DIR = original_artifacts_dir

class TestCFIntegration:
    """Integration tests for collaborative filtering models."""
    
    def test_als_with_sparse_data(self):
        """Test ALS with sparse interaction data."""
        # Create sparse interaction data
        sparse_interactions = pd.DataFrame({
            'user_id': ['1', '2', '3', '4', '5'],
            'item_id': ['A', 'B', 'C', 'D', 'E'],
            'weight': [5.0, 4.0, 3.0, 5.0, 4.0],
            'timestamp': pd.date_range('2020-01-01', periods=5)
        })
        
        model = ALSRecommender('test')
        model.fit(sparse_interactions)
        
        # Should handle sparse data gracefully
        assert model.model is not None
        assert len(model.user_mapping) == 5
        assert len(model.item_mapping) == 5
    
    def test_item_item_with_sparse_data(self):
        """Test Item-Item with sparse interaction data."""
        # Create sparse interaction data
        sparse_interactions = pd.DataFrame({
            'user_id': ['1', '2', '3', '4', '5'],
            'item_id': ['A', 'B', 'C', 'D', 'E'],
            'weight': [5.0, 4.0, 3.0, 5.0, 4.0],
            'timestamp': pd.date_range('2020-01-01', periods=5)
        })
        
        model = ItemItemRecommender('test')
        model.fit(sparse_interactions)
        
        # Should handle sparse data gracefully
        assert model.item_similarity_matrix is not None
        assert len(model.item_mapping) == 5
        assert len(model.user_mapping) == 5
    
    def test_als_edge_cases(self):
        """Test ALS with edge cases."""
        model = ALSRecommender('test')
        
        # Test with empty data
        empty_data = pd.DataFrame(columns=['user_id', 'item_id', 'weight', 'timestamp'])
        
        with pytest.raises(Exception):
            model.fit(empty_data)
        
        # Test with single user
        single_user_data = pd.DataFrame({
            'user_id': ['1', '1', '1'],
            'item_id': ['A', 'B', 'C'],
            'weight': [5.0, 4.0, 3.0],
            'timestamp': pd.date_range('2020-01-01', periods=3)
        })
        
        # Should handle single user gracefully
        model.fit(single_user_data)
        assert model.model is not None
    
    def test_item_item_edge_cases(self):
        """Test Item-Item with edge cases."""
        model = ItemItemRecommender('test')
        
        # Test with empty data
        empty_data = pd.DataFrame(columns=['user_id', 'item_id', 'weight', 'timestamp'])
        
        with pytest.raises(Exception):
            model.fit(empty_data)
        
        # Test with single item
        single_item_data = pd.DataFrame({
            'user_id': ['1', '2', '3'],
            'item_id': ['A', 'A', 'A'],
            'weight': [5.0, 4.0, 3.0],
            'timestamp': pd.date_range('2020-01-01', periods=3)
        })
        
        # Should handle single item gracefully
        model.fit(single_item_data)
        assert model.item_similarity_matrix is not None

if __name__ == "__main__":
    pytest.main([__file__]) 