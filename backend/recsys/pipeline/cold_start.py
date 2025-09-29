"""Cold start strategies for new users and items."""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Any
import structlog
from ..service.config import config
from ..service.cache import cache

logger = structlog.get_logger(__name__)

class ColdStartStrategy:
    """Cold start strategy manager."""
    
    def __init__(self, domain: str):
        """Initialize cold start strategy."""
        self.domain = domain
        self.popularity_items = []
        self.content_model = None
        self.user_interaction_counts = {}
        
    def set_popularity_items(self, popularity_items: List[Dict[str, Any]]):
        """Set popularity items for fallback recommendations."""
        self.popularity_items = popularity_items
        
    def set_content_model(self, content_model):
        """Set content model for content-based recommendations."""
        self.content_model = content_model
        
    def set_user_interaction_counts(self, interaction_counts: Dict[str, int]):
        """Set user interaction counts for alpha scheduling."""
        self.user_interaction_counts = interaction_counts

class NewUserStrategy:
    """Strategy for handling new users."""
    
    def __init__(self, cold_start_strategy: ColdStartStrategy):
        """Initialize new user strategy."""
        self.cold_start_strategy = cold_start_strategy
        
    def get_recommendations(self, user_id: str, k: int = 20) -> List[Dict[str, Any]]:
        """Get recommendations for a new user."""
        logger.info("Using new user strategy", user_id=user_id, domain=self.cold_start_strategy.domain)
        
        recommendations = []
        
        # 1. Popularity-based recommendations (60%)
        popular_count = int(k * 0.6)
        if self.cold_start_strategy.popularity_items:
            popular_recs = self.cold_start_strategy.popularity_items[:popular_count]
            recommendations.extend(popular_recs)
        
        # 2. Content-based diverse recommendations (40%)
        diverse_count = k - len(recommendations)
        if diverse_count > 0 and self.cold_start_strategy.content_model:
            diverse_recs = self._get_diverse_content_recommendations(diverse_count)
            recommendations.extend(diverse_recs)
        
        # 3. Fill remaining slots with more popular items
        if len(recommendations) < k and self.cold_start_strategy.popularity_items:
            remaining_count = k - len(recommendations)
            start_idx = len(recommendations)
            additional_recs = self.cold_start_strategy.popularity_items[start_idx:start_idx + remaining_count]
            recommendations.extend(additional_recs)
        
        return recommendations[:k]
    
    def _get_diverse_content_recommendations(self, k: int) -> List[Dict[str, Any]]:
        """Get diverse content-based recommendations using MMR."""
        if not self.cold_start_strategy.content_model:
            return []
        
        # Simplified MMR implementation
        # In practice, you'd implement proper Maximal Marginal Relevance
        
        # Get some random items for diversity (simplified approach)
        diverse_items = []
        
        # Sample from different genres/categories if available
        # For now, just return some popular items as diverse
        if self.cold_start_strategy.popularity_items:
            # Take every 5th item to ensure diversity
            diverse_items = self.cold_start_strategy.popularity_items[::5][:k]
        
        return diverse_items

class NewItemStrategy:
    """Strategy for handling new items."""
    
    def __init__(self, cold_start_strategy: ColdStartStrategy):
        """Initialize new item strategy."""
        self.cold_start_strategy = cold_start_strategy
        
    def get_recommendations(self, item_id: str, k: int = 20) -> List[Dict[str, Any]]:
        """Get recommendations for a new item."""
        logger.info("Using new item strategy", item_id=item_id, domain=self.cold_start_strategy.domain)
        
        recommendations = []
        
        # 1. Content-based similar items (70%)
        content_count = int(k * 0.7)
        if self.cold_start_strategy.content_model:
            content_recs = self.cold_start_strategy.content_model.get_similar_items(item_id, content_count)
            recommendations.extend(content_recs)
        
        # 2. Exploration items (30%)
        exploration_count = k - len(recommendations)
        if exploration_count > 0:
            exploration_recs = self._get_exploration_items(exploration_count)
            recommendations.extend(exploration_recs)
        
        return recommendations[:k]
    
    def _get_exploration_items(self, k: int) -> List[Dict[str, Any]]:
        """Get exploration items using ε-greedy strategy."""
        exploration_items = []
        
        # ε-greedy: 5% chance of showing random popular items
        epsilon = 0.05
        
        if np.random.random() < epsilon and self.cold_start_strategy.popularity_items:
            # Show random popular items for exploration
            random_indices = np.random.choice(
                len(self.cold_start_strategy.popularity_items), 
                size=min(k, len(self.cold_start_strategy.popularity_items)), 
                replace=False
            )
            exploration_items = [
                self.cold_start_strategy.popularity_items[i] 
                for i in random_indices
            ]
        
        return exploration_items

class AlphaScheduler:
    """Alpha scheduling for blending collaborative filtering and content-based recommendations."""
    
    def __init__(self, cold_start_strategy: ColdStartStrategy):
        """Initialize alpha scheduler."""
        self.cold_start_strategy = cold_start_strategy
        
    def get_alpha(self, user_id: str) -> float:
        """Get alpha value for blending based on user interaction count."""
        interaction_count = self.cold_start_strategy.user_interaction_counts.get(user_id, 0)
        
        if interaction_count < config.MIN_USER_INTERACTIONS:
            return config.ALPHA_MIN
        
        # Linear interpolation between min and max alpha
        alpha = config.ALPHA_MIN + (config.ALPHA_MAX - config.ALPHA_MIN) * (
            min(interaction_count / config.ALPHA_THRESHOLD, 1.0)
        )
        
        logger.debug("Alpha calculated", 
                    user_id=user_id, 
                    interaction_count=interaction_count, 
                    alpha=alpha)
        
        return alpha
    
    def blend_recommendations(self, 
                            cf_recommendations: List[Dict[str, Any]], 
                            content_recommendations: List[Dict[str, Any]], 
                            alpha: float) -> List[Dict[str, Any]]:
        """Blend CF and content recommendations using alpha."""
        if alpha >= 1.0:
            return cf_recommendations
        elif alpha <= 0.0:
            return content_recommendations
        
        # Create score mappings
        cf_scores = {rec['item_id']: rec['score'] for rec in cf_recommendations}
        content_scores = {rec['item_id']: rec['score'] for rec in content_recommendations}
        
        # Get all unique items
        all_items = set(cf_scores.keys()) | set(content_scores.keys())
        
        # Blend scores
        blended_recommendations = []
        for item_id in all_items:
            cf_score = cf_scores.get(item_id, 0.0)
            content_score = content_scores.get(item_id, 0.0)
            
            blended_score = alpha * cf_score + (1 - alpha) * content_score
            
            # Find original recommendation data
            original_rec = next((r for r in cf_recommendations if r['item_id'] == item_id), None)
            if not original_rec:
                original_rec = next((r for r in content_recommendations if r['item_id'] == item_id), None)
            
            if original_rec:
                blended_rec = original_rec.copy()
                blended_rec['score'] = blended_score
                blended_rec['source'] = 'blended'
                blended_rec['alpha'] = alpha
                blended_recommendations.append(blended_rec)
        
        # Sort by blended score
        blended_recommendations.sort(key=lambda x: x['score'], reverse=True)
        
        return blended_recommendations

class ColdStartManager:
    """Main cold start manager that coordinates all strategies."""
    
    def __init__(self, domain: str):
        """Initialize cold start manager."""
        self.domain = domain
        self.strategy = ColdStartStrategy(domain)
        self.new_user_strategy = NewUserStrategy(self.strategy)
        self.new_item_strategy = NewItemStrategy(self.strategy)
        self.alpha_scheduler = AlphaScheduler(self.strategy)
        
    def is_new_user(self, user_id: str) -> bool:
        """Check if user is new (cold start)."""
        interaction_count = self.strategy.user_interaction_counts.get(user_id, 0)
        return interaction_count < config.MIN_USER_INTERACTIONS
    
    def is_new_item(self, item_id: str) -> bool:
        """Check if item is new (cold start)."""
        # In practice, you'd check against a list of known items
        # For now, we'll assume items are new if they're not in popularity items
        known_items = {item['item_id'] for item in self.strategy.popularity_items}
        return item_id not in known_items
    
    def get_user_recommendations(self, user_id: str, k: int = 20) -> List[Dict[str, Any]]:
        """Get recommendations for a user, handling cold start if needed."""
        if self.is_new_user(user_id):
            logger.info("New user detected, using cold start strategy", user_id=user_id)
            return self.new_user_strategy.get_recommendations(user_id, k)
        else:
            # For existing users, alpha blending will be handled in the main pipeline
            logger.info("Existing user, using regular pipeline", user_id=user_id)
            return []
    
    def get_item_recommendations(self, item_id: str, k: int = 20) -> List[Dict[str, Any]]:
        """Get recommendations for an item, handling cold start if needed."""
        if self.is_new_item(item_id):
            logger.info("New item detected, using cold start strategy", item_id=item_id)
            return self.new_item_strategy.get_recommendations(item_id, k)
        else:
            # For existing items, use regular similarity search
            logger.info("Existing item, using regular pipeline", item_id=item_id)
            return []
    
    def get_alpha(self, user_id: str) -> float:
        """Get alpha value for blending."""
        return self.alpha_scheduler.get_alpha(user_id)
    
    def blend_recommendations(self, 
                            cf_recommendations: List[Dict[str, Any]], 
                            content_recommendations: List[Dict[str, Any]], 
                            user_id: str) -> List[Dict[str, Any]]:
        """Blend recommendations using alpha scheduling."""
        alpha = self.get_alpha(user_id)
        return self.alpha_scheduler.blend_recommendations(cf_recommendations, content_recommendations, alpha)
    
    def save_state(self):
        """Save cold start state to cache."""
        cache.set_user_features(self.domain, "cold_start_state", {
            'popularity_items': self.strategy.popularity_items,
            'user_interaction_counts': self.strategy.user_interaction_counts
        })
    
    def load_state(self):
        """Load cold start state from cache."""
        state = cache.get_user_features(self.domain, "cold_start_state")
        if state:
            self.strategy.popularity_items = state.get('popularity_items', [])
            self.strategy.user_interaction_counts = state.get('user_interaction_counts', {}) 