"""Ranking pipeline for scoring and ranking candidates."""

import time
from typing import Dict, List, Tuple, Optional, Any
import structlog
import pandas as pd
import numpy as np
from ..service.config import config
from ..service.cache import cache
from ..models.ranker_lgbm import LightGBMRanker, FeatureEngineer

logger = structlog.get_logger(__name__)

class RankingPipeline:
    """Ranking pipeline for scoring and ranking candidates."""
    
    def __init__(self, domain: str, ranker: LightGBMRanker, feature_engineer: FeatureEngineer):
        """Initialize ranking pipeline."""
        self.domain = domain
        self.ranker = ranker
        self.feature_engineer = feature_engineer
        
    def rank_candidates(self, 
                       user_id: str, 
                       candidates: List[Dict[str, Any]], 
                       als_model,
                       item_item_model,
                       content_model,
                       user_features: Dict[str, Any],
                       items_df: pd.DataFrame,
                       k: int = 20) -> List[Dict[str, Any]]:
        """Rank candidates using LightGBM model."""
        if not candidates:
            return []
        
        start_time = time.time()
        
        # Create features for ranking
        features_df = self.feature_engineer.create_ranking_features(
            user_id=user_id,
            candidates=candidates,
            als_model=als_model,
            item_item_model=item_item_model,
            content_model=content_model,
            user_features=user_features,
            items_df=items_df
        )
        
        # Predict ranking scores
        ranking_scores = self.ranker.predict(features_df)
        
        # Combine candidates with ranking scores
        ranked_candidates = []
        for i, candidate in enumerate(candidates):
            ranked_candidate = candidate.copy()
            ranked_candidate['ranking_score'] = float(ranking_scores[i])
            ranked_candidates.append(ranked_candidate)
        
        # Sort by ranking score
        ranked_candidates.sort(key=lambda x: x['ranking_score'], reverse=True)
        
        # Apply diversity and exploration
        final_candidates = self._apply_diversity_and_exploration(ranked_candidates, k)
        
        latency = (time.time() - start_time) * 1000
        cache.set_latency("ranking", latency)
        
        logger.info("Ranking completed", 
                   user_id=user_id, 
                   domain=self.domain,
                   candidates=len(candidates),
                   final_recommendations=len(final_candidates),
                   latency_ms=latency)
        
        return final_candidates
    
    def _apply_diversity_and_exploration(self, 
                                       ranked_candidates: List[Dict[str, Any]], 
                                       k: int) -> List[Dict[str, Any]]:
        """Apply diversity and exploration to final recommendations."""
        if len(ranked_candidates) <= k:
            return ranked_candidates
        
        # Simple diversity: ensure not too many items from the same source
        source_counts = {}
        final_candidates = []
        
        for candidate in ranked_candidates:
            source = candidate['source']
            current_count = source_counts.get(source, 0)
            
            # Limit items per source to ensure diversity
            max_per_source = max(1, k // 3)  # At most 1/3 from each source
            
            if current_count < max_per_source:
                final_candidates.append(candidate)
                source_counts[source] = current_count + 1
                
                if len(final_candidates) >= k:
                    break
        
        # If we still have slots, fill with remaining candidates
        if len(final_candidates) < k:
            remaining = [c for c in ranked_candidates if c not in final_candidates]
            final_candidates.extend(remaining[:k - len(final_candidates)])
        
        return final_candidates[:k]
    
    def get_feature_importance(self) -> Dict[str, float]:
        """Get feature importance from the ranking model."""
        return self.ranker.get_feature_importance()

class PopularityRanker:
    """Simple popularity-based ranking as baseline."""
    
    def __init__(self, popularity_items: List[Dict[str, Any]]):
        """Initialize popularity ranker."""
        self.popularity_items = popularity_items
    
    def rank_candidates(self, candidates: List[Dict[str, Any]], k: int = 20) -> List[Dict[str, Any]]:
        """Rank candidates by popularity."""
        # Create popularity score mapping
        popularity_scores = {item['item_id']: item['score'] for item in self.popularity_items}
        
        # Add popularity scores to candidates
        for candidate in candidates:
            item_id = candidate['item_id']
            candidate['popularity_score'] = popularity_scores.get(item_id, 0.0)
        
        # Sort by popularity score
        ranked_candidates = sorted(candidates, key=lambda x: x['popularity_score'], reverse=True)
        
        return ranked_candidates[:k]

class HybridRanker:
    """Hybrid ranking combining multiple approaches."""
    
    def __init__(self, 
                 ranking_pipeline: RankingPipeline,
                 popularity_ranker: PopularityRanker,
                 alpha: float = 0.7):
        """Initialize hybrid ranker."""
        self.ranking_pipeline = ranking_pipeline
        self.popularity_ranker = popularity_ranker
        self.alpha = alpha  # Weight for ML ranking vs popularity
    
    def rank_candidates(self, 
                       user_id: str, 
                       candidates: List[Dict[str, Any]], 
                       als_model,
                       item_item_model,
                       content_model,
                       user_features: Dict[str, Any],
                       items_df: pd.DataFrame,
                       k: int = 20) -> List[Dict[str, Any]]:
        """Rank candidates using hybrid approach."""
        if not candidates:
            return []
        
        # Get ML-based rankings
        ml_ranked = self.ranking_pipeline.rank_candidates(
            user_id, candidates, als_model, item_item_model, 
            content_model, user_features, items_df, k
        )
        
        # Get popularity-based rankings
        popularity_ranked = self.popularity_ranker.rank_candidates(candidates, k)
        
        # Blend rankings
        blended_candidates = self._blend_rankings(ml_ranked, popularity_ranked, k)
        
        return blended_candidates
    
    def _blend_rankings(self, 
                       ml_ranked: List[Dict[str, Any]], 
                       popularity_ranked: List[Dict[str, Any]], 
                       k: int) -> List[Dict[str, Any]]:
        """Blend ML and popularity rankings."""
        # Create score mappings
        ml_scores = {}
        for i, candidate in enumerate(ml_ranked):
            item_id = candidate['item_id']
            # Normalize ML score by position (1-based)
            ml_scores[item_id] = 1.0 / (i + 1)
        
        popularity_scores = {}
        for i, candidate in enumerate(popularity_ranked):
            item_id = candidate['item_id']
            # Normalize popularity score by position (1-based)
            popularity_scores[item_id] = 1.0 / (i + 1)
        
        # Get all unique items
        all_items = set(ml_scores.keys()) | set(popularity_scores.keys())
        
        # Blend scores
        blended_candidates = []
        for item_id in all_items:
            ml_score = ml_scores.get(item_id, 0.0)
            pop_score = popularity_scores.get(item_id, 0.0)
            
            blended_score = self.alpha * ml_score + (1 - self.alpha) * pop_score
            
            # Find original candidate data
            original_candidate = next((c for c in ml_ranked if c['item_id'] == item_id), None)
            if original_candidate:
                blended_candidate = original_candidate.copy()
                blended_candidate['blended_score'] = blended_score
                blended_candidates.append(blended_candidate)
        
        # Sort by blended score
        blended_candidates.sort(key=lambda x: x['blended_score'], reverse=True)
        
        return blended_candidates[:k]

class ColdStartRanker:
    """Ranking strategies for cold start scenarios."""
    
    def __init__(self, popularity_ranker: PopularityRanker, content_model):
        """Initialize cold start ranker."""
        self.popularity_ranker = popularity_ranker
        self.content_model = content_model
    
    def rank_new_user_candidates(self, 
                                candidates: List[Dict[str, Any]], 
                                k: int = 20) -> List[Dict[str, Any]]:
        """Rank candidates for new users."""
        # For new users, rely more on popularity and content diversity
        ranked_candidates = self.popularity_ranker.rank_candidates(candidates, k)
        
        # Apply diversity constraint
        diverse_candidates = self._apply_diversity_constraint(ranked_candidates, k)
        
        return diverse_candidates
    
    def rank_new_item_candidates(self, 
                                item_id: str,
                                candidates: List[Dict[str, Any]], 
                                k: int = 20) -> List[Dict[str, Any]]:
        """Rank candidates for new items."""
        # For new items, use content similarity + exploration
        if self.content_model:
            # Get content-based similar items
            content_similar = self.content_model.get_similar_items(item_id, k)
            
            # Blend with popularity
            blended = self._blend_content_and_popularity(content_similar, candidates, k)
            return blended
        
        # Fallback to popularity
        return self.popularity_ranker.rank_candidates(candidates, k)
    
    def _apply_diversity_constraint(self, 
                                  candidates: List[Dict[str, Any]], 
                                  k: int) -> List[Dict[str, Any]]:
        """Apply diversity constraint to ensure variety."""
        # Simple diversity: ensure different sources
        source_counts = {}
        diverse_candidates = []
        
        for candidate in candidates:
            source = candidate.get('source', 'unknown')
            current_count = source_counts.get(source, 0)
            
            # Limit items per source
            max_per_source = max(1, k // 4)
            
            if current_count < max_per_source:
                diverse_candidates.append(candidate)
                source_counts[source] = current_count + 1
                
                if len(diverse_candidates) >= k:
                    break
        
        # Fill remaining slots
        if len(diverse_candidates) < k:
            remaining = [c for c in candidates if c not in diverse_candidates]
            diverse_candidates.extend(remaining[:k - len(diverse_candidates)])
        
        return diverse_candidates[:k]
    
    def _blend_content_and_popularity(self, 
                                    content_candidates: List[Dict[str, Any]], 
                                    popularity_candidates: List[Dict[str, Any]], 
                                    k: int) -> List[Dict[str, Any]]:
        """Blend content-based and popularity-based candidates."""
        # Weight content similarity higher for new items
        content_weight = 0.8
        popularity_weight = 0.2
        
        # Create score mappings
        content_scores = {c['item_id']: c['score'] for c in content_candidates}
        popularity_scores = {c['item_id']: c['score'] for c in popularity_candidates}
        
        # Get all unique items
        all_items = set(content_scores.keys()) | set(popularity_scores.keys())
        
        # Blend scores
        blended_candidates = []
        for item_id in all_items:
            content_score = content_scores.get(item_id, 0.0)
            pop_score = popularity_scores.get(item_id, 0.0)
            
            blended_score = content_weight * content_score + popularity_weight * pop_score
            
            # Find original candidate data
            original_candidate = next((c for c in content_candidates if c['item_id'] == item_id), None)
            if original_candidate:
                blended_candidate = original_candidate.copy()
                blended_candidate['blended_score'] = blended_score
                blended_candidates.append(blended_candidate)
        
        # Sort by blended score
        blended_candidates.sort(key=lambda x: x['blended_score'], reverse=True)
        
        return blended_candidates[:k] 