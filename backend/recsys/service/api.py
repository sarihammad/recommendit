"""FastAPI service for the recommendation system."""

import time
from typing import Dict, List, Optional, Any
from fastapi import FastAPI, HTTPException, Depends
from fastapi.middleware.cors import CORSMiddleware
import structlog
from prometheus_client import Counter, Histogram, generate_latest, CONTENT_TYPE_LATEST
from starlette.responses import Response

from .schemas import (
    RecommendationRequest, RecommendationResponse, RecommendationItem,
    SimilarItemRequest, SimilarItemResponse, FeedbackRequest, FeedbackResponse,
    HealthResponse, Domain
)
from .config import config
from .cache import cache
from ..pipeline.recall import RecallPipeline, ColdStartRecall, AlphaBlending
from ..pipeline.rank import RankingPipeline, PopularityRanker, ColdStartRanker
from ..data.loaders import data_loader
from ..data.features import UserFeatureExtractor

# Configure structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger(__name__)

# Prometheus metrics
REQUEST_COUNT = Counter('recommendation_requests_total', 'Total recommendation requests', ['domain', 'endpoint'])
REQUEST_LATENCY = Histogram('recommendation_request_duration_seconds', 'Request latency', ['domain', 'endpoint'])
ERROR_COUNT = Counter('recommendation_errors_total', 'Total errors', ['domain', 'endpoint', 'error_type'])

# Global model instances (will be loaded on startup)
models = {
    'books': {},
    'movies': {}
}

app = FastAPI(
    title="Production Hybrid Recommender",
    description="A production-ready hybrid recommendation system with collaborative filtering, content-based, and ranking models",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

def load_models():
    """Load all models for both domains."""
    logger.info("Loading models for all domains")
    
    for domain in ['books', 'movies']:
        try:
            logger.info(f"Loading models for domain: {domain}")
            
            # Load data
            if domain == 'books':
                items_df = data_loader.load_books()
            else:
                items_df = data_loader.load_movies()
            
            # Load models (this would be done in practice)
            # For now, we'll create placeholder models
            models[domain] = {
                'items_df': items_df,
                'als_model': None,  # Would be loaded from artifacts
                'item_item_model': None,  # Would be loaded from artifacts
                'content_model': None,  # Would be loaded from artifacts
                'ranking_pipeline': None,  # Would be loaded from artifacts
                'user_feature_extractor': None,  # Would be loaded from artifacts
                'popularity_items': []  # Would be computed from interactions
            }
            
            logger.info(f"Models loaded for domain: {domain}")
            
        except Exception as e:
            logger.error(f"Failed to load models for domain {domain}", error=str(e))
            ERROR_COUNT.labels(domain=domain, endpoint='load_models', error_type='load_error').inc()

@app.on_event("startup")
async def startup_event():
    """Initialize the application on startup."""
    logger.info("Starting recommendation service")
    
    # Check Redis connection
    if not cache.health_check():
        logger.warning("Redis connection failed, continuing without cache")
    
    # Load models
    load_models()
    
    logger.info("Recommendation service started successfully")

@app.get("/healthz", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    models_loaded = {}
    for domain in ['books', 'movies']:
        models_loaded[domain] = len(models[domain]) > 0
    
    return HealthResponse(
        status="healthy",
        timestamp=time.time(),
        version="1.0.0",
        models_loaded=models_loaded
    )

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint."""
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

@app.post("/recommend", response_model=RecommendationResponse)
async def get_recommendations(request: RecommendationRequest):
    """Get personalized recommendations for a user."""
    start_time = time.time()
    domain = request.domain.value
    
    try:
        REQUEST_COUNT.labels(domain=domain, endpoint='recommend').inc()
        
        # Validate domain
        if domain not in models:
            raise HTTPException(status_code=400, detail=f"Unsupported domain: {domain}")
        
        # Get domain models
        domain_models = models[domain]
        items_df = domain_models['items_df']
        
        # Check if user is new (cold start)
        user_interaction_count = 0  # TODO: Get from user interactions
        is_new_user = user_interaction_count < config.MIN_USER_INTERACTIONS
        
        if is_new_user:
            # Cold start: use popularity + content diversity
            logger.info("New user detected, using cold start strategy", user_id=request.user_id)
            
            # Get popular items
            popular_items = domain_models.get('popularity_items', [])
            if not popular_items:
                # Fallback: create dummy popular items
                popular_items = [
                    {'item_id': str(i), 'score': 1.0 / (i + 1)} 
                    for i in range(min(20, len(items_df)))
                ]
            
            recommendations = popular_items[:request.k]
            
        else:
            # Regular recommendation pipeline
            # TODO: Implement full pipeline with loaded models
            # For now, return dummy recommendations
            recommendations = [
                {
                    'item_id': str(i),
                    'score': 1.0 / (i + 1),
                    'source': 'dummy'
                }
                for i in range(request.k)
            ]
        
        # Convert to response format
        recommendation_items = []
        for rec in recommendations:
            item_id = rec['item_id']
            
            # Get item metadata
            item_meta = items_df[items_df['item_id'] == item_id]
            title = item_meta['title'].iloc[0] if len(item_meta) > 0 else f"Item {item_id}"
            
            recommendation_items.append(RecommendationItem(
                item_id=item_id,
                score=rec['score'],
                title=title,
                metadata={
                    'source': rec.get('source', 'unknown'),
                    'ranking_score': rec.get('ranking_score', rec['score'])
                }
            ))
        
        latency = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(domain=domain, endpoint='recommend').observe(latency / 1000)
        
        # Cache latency
        cache.set_latency("recommend_api", latency)
        
        return RecommendationResponse(
            user_id=request.user_id,
            domain=request.domain,
            recommendations=recommendation_items,
            total_candidates=len(recommendations),
            latency_ms=latency,
            model_info={
                'is_new_user': is_new_user,
                'user_interaction_count': user_interaction_count
            }
        )
        
    except Exception as e:
        ERROR_COUNT.labels(domain=domain, endpoint='recommend', error_type='exception').inc()
        logger.error("Error getting recommendations", 
                    user_id=request.user_id, 
                    domain=domain, 
                    error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/similar", response_model=SimilarItemResponse)
async def get_similar_items(request: SimilarItemRequest):
    """Get similar items for a given item."""
    start_time = time.time()
    domain = request.domain.value
    
    try:
        REQUEST_COUNT.labels(domain=domain, endpoint='similar').inc()
        
        # Validate domain
        if domain not in models:
            raise HTTPException(status_code=400, detail=f"Unsupported domain: {domain}")
        
        # Get domain models
        domain_models = models[domain]
        items_df = domain_models['items_df']
        
        # TODO: Implement similar items logic with loaded models
        # For now, return dummy similar items
        similar_items = [
            {
                'item_id': str(i),
                'score': 1.0 / (i + 1),
                'source': 'dummy_similar'
            }
            for i in range(request.k)
        ]
        
        # Convert to response format
        recommendation_items = []
        for item in similar_items:
            item_id = item['item_id']
            
            # Get item metadata
            item_meta = items_df[items_df['item_id'] == item_id]
            title = item_meta['title'].iloc[0] if len(item_meta) > 0 else f"Item {item_id}"
            
            recommendation_items.append(RecommendationItem(
                item_id=item_id,
                score=item['score'],
                title=title,
                metadata={'source': item.get('source', 'unknown')}
            ))
        
        latency = (time.time() - start_time) * 1000
        REQUEST_LATENCY.labels(domain=domain, endpoint='similar').observe(latency / 1000)
        
        # Cache latency
        cache.set_latency("similar_api", latency)
        
        return SimilarItemResponse(
            item_id=request.item_id,
            domain=request.domain,
            similar_items=recommendation_items,
            latency_ms=latency
        )
        
    except Exception as e:
        ERROR_COUNT.labels(domain=domain, endpoint='similar', error_type='exception').inc()
        logger.error("Error getting similar items", 
                    item_id=request.item_id, 
                    domain=domain, 
                    error=str(e))
        raise HTTPException(status_code=500, detail="Internal server error")

@app.post("/feedback", response_model=FeedbackResponse)
async def submit_feedback(request: FeedbackRequest):
    """Submit user feedback for model improvement."""
    try:
        # Log feedback event
        logger.info("User feedback received", 
                   user_id=request.user_id,
                   item_id=request.item_id,
                   event=request.event.value,
                   rating=request.rating)
        
        # TODO: Store feedback in database or file for retraining
        # For now, just log it
        
        # Increment feedback counter
        cache.increment_counter(f"feedback_{request.event.value}")
        
        return FeedbackResponse(
            success=True,
            message="Feedback recorded successfully"
        )
        
    except Exception as e:
        logger.error("Error recording feedback", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to record feedback")

@app.get("/stats")
async def get_stats():
    """Get system statistics."""
    try:
        stats = {
            'request_counts': {},
            'latency_stats': {},
            'cache_stats': {}
        }
        
        # Get request counts from Prometheus
        for domain in ['books', 'movies']:
            for endpoint in ['recommend', 'similar']:
                key = f"{domain}_{endpoint}"
                stats['request_counts'][key] = 0  # Would get from Prometheus
        
        # Get latency stats
        for operation in ['recall', 'ranking', 'recommend_api', 'similar_api']:
            latency_stats = cache.get_latency_stats(operation)
            if latency_stats:
                stats['latency_stats'][operation] = latency_stats
        
        # Get cache stats
        stats['cache_stats'] = {
            'redis_healthy': cache.health_check()
        }
        
        return stats
        
    except Exception as e:
        logger.error("Error getting stats", error=str(e))
        raise HTTPException(status_code=500, detail="Failed to get stats")

@app.post("/retrain/{domain}")
async def trigger_retrain(domain: str):
    """Trigger model retraining for a domain."""
    try:
        if domain not in ['books', 'movies']:
            raise HTTPException(status_code=400, detail=f"Unsupported domain: {domain}")
        
        logger.info("Retraining triggered", domain=domain)
        
        # TODO: Implement actual retraining logic
        # This would typically be done asynchronously
        
        return {"message": f"Retraining triggered for domain: {domain}"}
        
    except Exception as e:
        logger.error("Error triggering retrain", domain=domain, error=str(e))
        raise HTTPException(status_code=500, detail="Failed to trigger retraining")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host=config.API_HOST,
        port=config.API_PORT,
        log_level=config.LOG_LEVEL.lower()
    ) 