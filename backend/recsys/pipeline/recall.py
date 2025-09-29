"""Recall pipeline for candidate generation."""

import time
from typing import Dict, List, Tuple, Optional, Any
import structlog
from ..service.config import config
from ..service.cache import cache

logger = structlog.get_logger(__name__)

class RecallPipeline:
    """Recall pipeline for candidate generation."""
    
    def __init__(self, domain: str, als_model, item_item_model, content_model):
        """Initialize recall pipeline."""
        self.domain = domain
        self.als_model = als_model
        self.item_item_model = item_item_model
        self.content_model = content_model
        
    def get_candidates(self, user_id: str, n_candidates: int = None) -> List[Dict[str, Any]]:
        """Get candidates for a user from all sources."""
        if n_candidates is None:
            n_candidates = config.RECALL_CANDIDATES
        
        start_time = time.time()
        
        # Check cache first
        cached_candidates = cache.get_user_candidates(self.domain, user_id)
        if cached_candidates:
            logger.info("Using cached candidates", user_id=user_id, domain=self.domain)
            return cached_candidates[:n_candidates]
        
        # Get candidates from different sources
        candidates = []
        
        # ALS candidates
        als_candidates = self._get_als_candidates(user_id)
        candidates.extend(als_candidates)
        
        # Item-item candidates
        item_item_candidates = self._get_item_item_candidates(user_id)
        candidates.extend(item_item_candidates)
        
        # Content candidates
        content_candidates = self._get_content_candidates(user_id)
        candidates.extend(content_candidates)
        
        # Deduplicate and rank
        final_candidates = self._deduplicate_candidates(candidates, n_candidates)
        
        # Cache results
        cache.set_user_candidates(self.domain, user_id, final_candidates)
        
        latency = (time.time() - start_time) * 1000
        cache.set_latency("recall", latency)
        
        logger.info("Recall completed", 
                   user_id=user_id, 
                   domain=self.domain,
                   total_candidates=len(candidates),
                   final_candidates=len(final_candidates),
                   latency_ms=latency)
        
        return final_candidates
    
    def _get_als_candidates(self, user_id: str) -> List[Dict[str, Any]]:
        """Get candidates from ALS model."""
        try:
            if self.als_model:
                max_candidates = config.MAX_CANDIDATES_PER_SOURCE
                candidates = self.als_model.get_user_candidates(user_id, max_candidates)
                logger.debug("ALS candidates", user_id=user_id, count=len(candidates))
                return candidates
        except Exception as e:
            logger.error("Error getting ALS candidates", user_id=user_id, error=str(e))
        
        return []
    
    def _get_item_item_candidates(self, user_id: str) -> List[Dict[str, Any]]:
        """Get candidates from item-item model."""
        try:
            if self.item_item_model:
                max_candidates = config.MAX_CANDIDATES_PER_SOURCE
                candidates = self.item_item_model.get_user_candidates(user_id, max_candidates)
                logger.debug("Item-item candidates", user_id=user_id, count=len(candidates))
                return candidates
        except Exception as e:
            logger.error("Error getting item-item candidates", user_id=user_id, error=str(e))
        
        return []
    
    def _get_content_candidates(self, user_id: str) -> List[Dict[str, Any]]:
        """Get candidates from content model."""
        try:
            if self.content_model:
                # For content-based, we need user history
                # This is a simplified version - in practice, you'd get user history from interactions
                user_history = []  # TODO: Get actual user history
                max_candidates = config.MAX_CANDIDATES_PER_SOURCE
                candidates = self.content_model.get_user_candidates(user_id, user_history, max_candidates)
                logger.debug("Content candidates", user_id=user_id, count=len(candidates))
                return candidates
        except Exception as e:
            logger.error("Error getting content candidates", user_id=user_id, error=str(e))
        
        return []
    
    def _deduplicate_candidates(self, candidates: List[Dict[str, Any]], n_candidates: int) -> List[Dict[str, Any]]:
        """Deduplicate candidates and return top-N."""
        # Create a dictionary to track unique items
        unique_candidates = {}
        
        for candidate in candidates:
            item_id = candidate['item_id']
            score = candidate['score']
            source = candidate['source']
            
            if item_id in unique_candidates:
                # If item already exists, take the higher score
                if score > unique_candidates[item_id]['score']:
                    unique_candidates[item_id] = candidate
            else:
                unique_candidates[item_id] = candidate
        
        # Sort by score and return top-N
        sorted_candidates = sorted(
            unique_candidates.values(), 
            key=lambda x: x['score'], 
            reverse=True
        )
        
        return sorted_candidates[:n_candidates]
    
    def get_similar_items(self, item_id: str, n_similar: int = 20) -> List[Dict[str, Any]]:
        """Get similar items for a given item."""
        start_time = time.time()
        
        candidates = []
        
        # Get similar items from different sources
        try:
            if self.als_model:
                als_similar = self.als_model.get_similar_items(item_id, n_similar)
                candidates.extend(als_similar)
        except Exception as e:
            logger.error("Error getting ALS similar items", item_id=item_id, error=str(e))
        
        try:
            if self.item_item_model:
                item_item_similar = self.item_item_model.get_similar_items(item_id, n_similar)
                candidates.extend(item_item_similar)
        except Exception as e:
            logger.error("Error getting item-item similar items", item_id=item_id, error=str(e))
        
        try:
            if self.content_model:
                content_similar = self.content_model.get_similar_items(item_id, n_similar)
                candidates.extend(content_similar)
        except Exception as e:
            logger.error("Error getting content similar items", item_id=item_id, error=str(e))
        
        # Deduplicate and rank
        final_candidates = self._deduplicate_candidates(candidates, n_similar)
        
        latency = (time.time() - start_time) * 1000
        cache.set_latency("similar_items", latency)
        
        logger.info("Similar items retrieved", 
                   item_id=item_id, 
                   domain=self.domain,
                   candidates=len(final_candidates),
                   latency_ms=latency)
        
        return final_candidates

class ColdStartRecall:
    """Cold start recall strategies."""
    
    def __init__(self, domain: str, content_model, popularity_items: List[Dict[str, Any]]):
        """Initialize cold start recall."""
        self.domain = domain
        self.content_model = content_model
        self.popularity_items = popularity_items
    
    def get_new_user_candidates(self, n_candidates: int = 20) -> List[Dict[str, Any]]:
        """Get candidates for new users (content + popularity)."""
        candidates = []
        
        # Add popular items
        popular_count = min(n_candidates // 2, len(self.popularity_items))
        candidates.extend(self.popularity_items[:popular_count])
        
        # Add diverse content-based items
        if self.content_model and len(candidates) < n_candidates:
            remaining_count = n_candidates - len(candidates)
            # Get diverse content candidates (simplified - in practice use MMR)
            content_candidates = self._get_diverse_content_candidates(remaining_count)
            candidates.extend(content_candidates)
        
        return candidates[:n_candidates]
    
    def get_new_item_candidates(self, item_id: str, n_candidates: int = 20) -> List[Dict[str, Any]]:
        """Get candidates for new items (content similarity + exploration)."""
        candidates = []
        
        # Get content-based similar items
        if self.content_model:
            content_similar = self.content_model.get_similar_items(item_id, n_candidates)
            candidates.extend(content_similar)
        
        # Add some exploration (popular items)
        if len(candidates) < n_candidates:
            remaining_count = n_candidates - len(candidates)
            exploration_count = min(remaining_count, len(self.popularity_items) // 4)
            candidates.extend(self.popularity_items[:exploration_count])
        
        return candidates[:n_candidates]
    
    def _get_diverse_content_candidates(self, n_candidates: int) -> List[Dict[str, Any]]:
        """Get diverse content-based candidates (simplified MMR)."""
        # This is a simplified version of Maximal Marginal Relevance
        # In practice, you'd implement proper MMR with diversity penalty
        
        candidates = []
        # For now, just return some popular items as diverse candidates
        candidates.extend(self.popularity_items[:n_candidates])
        
        return candidates

class AlphaBlending:
    """Alpha blending for cold start handling."""
    
    @staticmethod
    def get_alpha(user_interaction_count: int) -> float:
        """Get alpha value for blending based on user interaction count."""
        if user_interaction_count < config.MIN_USER_INTERACTIONS:
            return config.ALPHA_MIN
        
        # Linear interpolation between min and max alpha
        alpha = config.ALPHA_MIN + (config.ALPHA_MAX - config.ALPHA_MIN) * (
            min(user_interaction_count / config.ALPHA_THRESHOLD, 1.0)
        )
        
        return alpha
    
    @staticmethod
    def blend_candidates(cf_candidates: List[Dict[str, Any]], 
                        content_candidates: List[Dict[str, Any]], 
                        alpha: float) -> List[Dict[str, Any]]:
        """Blend CF and content candidates using alpha."""
        if alpha >= 1.0:
            return cf_candidates
        elif alpha <= 0.0:
            return content_candidates
        
        # Create item score mapping
        cf_scores = {c['item_id']: c['score'] for c in cf_candidates}
        content_scores = {c['item_id']: c['score'] for c in content_candidates}
        
        # Get all unique items
        all_items = set(cf_scores.keys()) | set(content_scores.keys())
        
        # Blend scores
        blended_candidates = []
        for item_id in all_items:
            cf_score = cf_scores.get(item_id, 0.0)
            content_score = content_scores.get(item_id, 0.0)
            
            blended_score = alpha * cf_score + (1 - alpha) * content_score
            
            blended_candidates.append({
                'item_id': item_id,
                'score': blended_score,
                'source': 'blended'
            })
        
        # Sort by blended score
        blended_candidates.sort(key=lambda x: x['score'], reverse=True)
        
        return blended_candidates 