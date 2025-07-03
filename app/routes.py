from fastapi import APIRouter, HTTPException
from app.fuzzy_matcher import best_match_by_edit_distance
from app.recommender import Recommender
from app.data_loader import load_movie_data
from app.models import RecommendationRequest

router = APIRouter()
movie_data = load_movie_data()
recommender = Recommender(movie_data)
titles = movie_data["title"].tolist()

@router.post("/recommend")
def recommend_movie(request: RecommendationRequest):
    """
    Recommend movies based on a given title, with optional genre and year filters.
    """
    match = best_match_by_edit_distance(request.title, titles)
    if not match:
        raise HTTPException(status_code=404, detail="Movie title not found")

    results = recommender.recommend(match, genre=request.genre, year=request.year)
    return {"title": match, "recommendations": results}

