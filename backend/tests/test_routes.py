from fastapi.testclient import TestClient
from app.main import app

client = TestClient(app)

def test_recommend_exact_title():
    """
    Test that a known movie title returns valid recommendations.
    """
    response = client.post("/recommend", json={"title": "Inception"})
    assert response.status_code == 200
    data = response.json()
    assert "recommendations" in data
    assert isinstance(data["recommendations"], list)

def test_recommend_fuzzy_match():
    """
    Test that a slightly misspelled title is correctly matched using fuzzy logic.
    """
    response = client.post("/recommend", json={"title": "Incepton"})
    assert response.status_code == 200
    data = response.json()
    assert data["title"].lower() == "inception"
    assert len(data["recommendations"]) > 0

def test_recommend_not_found():
    """
    Test that an unknown title returns a 404 error with appropriate message.
    """
    response = client.post("/recommend", json={"title": "asdjklqwe123"})
    assert response.status_code == 404
    assert response.json()["detail"] == 'Movie title not found'