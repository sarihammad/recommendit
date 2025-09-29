# RecommendIt

A hybrid recommendation system that combines collaborative filtering, content-based, and ranking models for both books and movies domains.

## Features

- **Two-Stage Pipeline**: Candidate generation (recall) + ranking
- **Multiple Models**: ALS, Item-Item CF, SBERT embeddings, LightGBM ranking
- **Cold Start Handling**: Robust strategies for new users and items
- **Production APIs**: FastAPI with Redis caching and Prometheus metrics
- **Performance Optimized**: Sub-150ms end-to-end latency targets
- **Docker Ready**: Complete containerization with docker-compose

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   FastAPI       │    │   Redis Cache   │    │   Prometheus    │
│   Service       │◄──►│   (Features &   │    │   Metrics       │
│                 │    │   Candidates)   │    │                 │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Recall        │    ┌─────────────────┐
│   Pipeline      │───►│   ALS Model     │
│                 │    └─────────────────┘
│   (Candidate    │    ┌─────────────────┐
│   Generation)   │───►│ Item-Item CF    │
│                 │    └─────────────────┘
│                 │    ┌─────────────────┐
│                 │───►│ SBERT + Faiss   │
└─────────────────┘    └─────────────────┘
         │
         ▼
┌─────────────────┐
│   Ranking       │    ┌─────────────────┐
│   Pipeline      │───►│ LightGBM        │
│                 │    │ Ranker          │
│   (Scoring &    │    └─────────────────┘
│   Reordering)   │
└─────────────────┘
```

## Quick Start

### Prerequisites

- Python 3.11+
- Docker and docker-compose
- Redis (optional, for caching)

### Installation

1. **Clone the repository**

   ```bash
   git clone <repository-url>
   cd recommendit
   ```

2. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

3. **Train models for books domain**

   ```bash
   python recsys/scripts/train_books.py
   ```

4. **Start the service with Docker**
   ```bash
   docker-compose up --build
   ```

### Local Development

1. **Start Redis** (if not using Docker)

   ```bash
   redis-server
   ```

2. **Run the API service**

   ```bash
   python -m recsys.service.api
   ```

3. **Access the API**
   - API Documentation: http://localhost:8000/docs
   - Health Check: http://localhost:8000/healthz
   - Metrics: http://localhost:8000/metrics

## API Endpoints

### Get Recommendations

```bash
curl -X POST "http://localhost:8000/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "12345",
    "domain": "books",
    "k": 20
  }'
```

**Response:**

```json
{
  "user_id": "12345",
  "domain": "books",
  "recommendations": [
    {
      "item_id": "9780439023481",
      "score": 0.95,
      "title": "The Hunger Games",
      "metadata": {
        "source": "als",
        "ranking_score": 0.92
      }
    }
  ],
  "total_candidates": 1000,
  "latency_ms": 45.2,
  "model_info": {
    "is_new_user": false,
    "user_interaction_count": 15
  }
}
```

### Get Similar Items

```bash
curl -X POST "http://localhost:8000/similar" \
  -H "Content-Type: application/json" \
  -d '{
    "item_id": "9780439023481",
    "domain": "books",
    "k": 10
  }'
```

### Submit Feedback

```bash
curl -X POST "http://localhost:8000/feedback" \
  -H "Content-Type: application/json" \
  -d '{
    "user_id": "12345",
    "item_id": "9780439023481",
    "event": "click",
    "rating": 4.5
  }'
```

## Training Pipeline

### Books Domain Training

```bash
# Train all models for books
python recsys/scripts/train_books.py
```

The training pipeline includes:

1. **Data Loading & Preprocessing**

   - Schema inference and standardization
   - Time-based train/validation/test split
   - Implicit weight generation from ratings

2. **Content Embedding Model**

   - SBERT embeddings for titles and descriptions
   - Faiss index for fast similarity search
   - Genre and metadata features

3. **Collaborative Filtering Models**

   - ALS (Alternating Least Squares) for implicit feedback
   - Item-Item similarity matrix
   - User-Item interaction matrix

4. **Feature Engineering**

   - Content-based features (TF-IDF, genres, years)
   - User features (interaction counts, preferences)
   - Context features (source, popularity)

5. **LightGBM Ranking Model**
   - LambdaRank objective for ranking
   - Feature importance analysis
   - Model persistence

### Movies Domain Training

```bash
# Train all models for movies (similar to books)
python recsys/scripts/train_movies.py
```

## Configuration

Environment variables can be set to configure the system:

```bash
# Model parameters
RECSYS_ALS_FACTORS=100
RECSYS_ALS_ITERATIONS=20
RECSYS_SBERT_MODEL=all-MiniLM-L6-v2

# Performance targets
RECSYS_RECALL_LATENCY_P95_MS=60
RECSYS_RANK_LATENCY_P95_MS=100
RECSYS_E2E_LATENCY_P95_MS=150

# Redis configuration
RECSYS_REDIS_HOST=localhost
RECSYS_REDIS_PORT=6379
RECSYS_REDIS_TTL=3600

# API configuration
RECSYS_API_HOST=0.0.0.0
RECSYS_API_PORT=8000
RECSYS_LOG_LEVEL=INFO
```

## Performance Targets

- **Recall Stage**: < 60ms p95 latency
- **Ranking Stage**: < 100ms p95 latency
- **End-to-End**: < 150ms p95 latency
- **Throughput**: 1000+ requests/second
- **Cache Hit Rate**: > 80% for hot users

## Monitoring & Observability

### Metrics

The system exposes Prometheus metrics at `/metrics`:

- Request counts by domain and endpoint
- Latency histograms
- Error rates
- Cache hit rates
- Model prediction latencies

### Health Checks

- Service health: `GET /healthz`
- Redis connectivity
- Model loading status
- Artifact availability

### Logging

Structured logging with JSON format:

```json
{
  "timestamp": "2024-01-15T10:30:00Z",
  "level": "info",
  "user_id": "12345",
  "domain": "books",
  "latency_ms": 45.2,
  "candidates": 1000,
  "recommendations": 20
}
```

## Cold Start Strategies

### New Users

- Content-based recommendations using item metadata
- Popularity-based fallback
- Diversity constraints (MMR)
- Alpha blending as interactions accumulate

### New Items

- Content similarity using SBERT embeddings
- Exploration with ε-greedy strategy
- Exposure through popular item recommendations

## Model Artifacts

Trained models are saved in `artifacts/{domain}/`:

```
artifacts/
├── books/
│   ├── als_model.pkl
│   ├── als_factors.npz
│   ├── item_similarity_matrix.npy
│   ├── faiss.index
│   ├── item_embeddings.npy
│   ├── lgbm_model.txt
│   ├── feature_processors.pkl
│   └── popularity_items.pkl
└── movies/
    └── ...
```

## Development

### Project Structure

```
recsys/
├── data/              # Data loading and preprocessing
├── models/            # ML models (ALS, CF, SBERT, LightGBM)
├── pipeline/          # Recall and ranking pipelines
├── service/           # FastAPI service and utilities
└── scripts/           # Training and evaluation scripts
```

### Running Tests

```bash
pytest tests/
```

### Code Quality

```bash
# Type checking
mypy recsys/

# Linting
flake8 recsys/

# Formatting
black recsys/
```

## Deployment

### Docker Deployment

```bash
# Build and run with docker-compose
docker-compose up --build -d

# Scale the service
docker-compose up --scale recommender=3
```

### Kubernetes Deployment

```yaml
apiVersion: apps/v1
kind: Deployment
metadata:
  name: recommender
spec:
  replicas: 3
  selector:
    matchLabels:
      app: recommender
  template:
    metadata:
      labels:
        app: recommender
    spec:
      containers:
        - name: recommender
          image: recommender:latest
          ports:
            - containerPort: 8000
          env:
            - name: RECSYS_REDIS_HOST
              value: "redis-service"
```

## Evaluation

### Offline Metrics

- **Recall@K**: Fraction of relevant items in top-K
- **NDCG@K**: Normalized Discounted Cumulative Gain
- **MAP@K**: Mean Average Precision

### A/B Testing

- Compare popularity baseline vs hybrid model
- Measure user engagement metrics
- Monitor business KPIs

## Contributing

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Ensure all tests pass
5. Submit a pull request

## License

MIT License - see LICENSE file for details.

## Support

For questions and support:

- Create an issue on GitHub
- Check the API documentation at `/docs`
- Review the training logs for debugging
