"""Pydantic schemas for API requests and responses."""

from typing import List, Optional, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime
from enum import Enum

class Domain(str, Enum):
    """Supported domains."""
    BOOKS = "books"
    MOVIES = "movies"

class EventType(str, Enum):
    """User interaction event types."""
    VIEW = "view"
    CLICK = "click"
    ADD = "add"
    PURCHASE = "purchase"
    RATE = "rate"

class RecommendationRequest(BaseModel):
    """Request for recommendations."""
    user_id: str = Field(..., description="User ID")
    domain: Domain = Field(..., description="Domain (books or movies)")
    k: int = Field(default=20, ge=1, le=100, description="Number of recommendations")
    
class SimilarItemRequest(BaseModel):
    """Request for similar items."""
    item_id: str = Field(..., description="Item ID")
    domain: Domain = Field(..., description="Domain (books or movies)")
    k: int = Field(default=20, ge=1, le=100, description="Number of similar items")

class FeedbackRequest(BaseModel):
    """User feedback event."""
    user_id: str = Field(..., description="User ID")
    item_id: str = Field(..., description="Item ID")
    event: EventType = Field(..., description="Event type")
    timestamp: Optional[datetime] = Field(default_factory=datetime.now, description="Event timestamp")
    rating: Optional[float] = Field(None, ge=0, le=5, description="Rating if applicable")
    metadata: Optional[Dict[str, Any]] = Field(default=None, description="Additional metadata")

class RecommendationItem(BaseModel):
    """Individual recommendation item."""
    item_id: str = Field(..., description="Item ID")
    score: float = Field(..., description="Recommendation score")
    title: str = Field(..., description="Item title")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Additional metadata")

class RecommendationResponse(BaseModel):
    """Response for recommendations."""
    user_id: str = Field(..., description="User ID")
    domain: Domain = Field(..., description="Domain")
    recommendations: List[RecommendationItem] = Field(..., description="List of recommendations")
    total_candidates: int = Field(..., description="Total candidates considered")
    latency_ms: float = Field(..., description="Request latency in milliseconds")
    model_info: Dict[str, Any] = Field(default_factory=dict, description="Model information")

class SimilarItemResponse(BaseModel):
    """Response for similar items."""
    item_id: str = Field(..., description="Query item ID")
    domain: Domain = Field(..., description="Domain")
    similar_items: List[RecommendationItem] = Field(..., description="List of similar items")
    latency_ms: float = Field(..., description="Request latency in milliseconds")

class FeedbackResponse(BaseModel):
    """Response for feedback submission."""
    success: bool = Field(..., description="Whether feedback was recorded successfully")
    message: str = Field(..., description="Response message")

class HealthResponse(BaseModel):
    """Health check response."""
    status: str = Field(..., description="Service status")
    timestamp: datetime = Field(..., description="Current timestamp")
    version: str = Field(..., description="Service version")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")

class ErrorResponse(BaseModel):
    """Error response."""
    error: str = Field(..., description="Error message")
    detail: Optional[str] = Field(None, description="Error details")
    timestamp: datetime = Field(default_factory=datetime.now, description="Error timestamp") 