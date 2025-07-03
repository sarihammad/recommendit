from typing import Optional
from pydantic import BaseModel

class RecommendationRequest(BaseModel):
    title: str
    genre: Optional[str] = None
    year: Optional[int] = None