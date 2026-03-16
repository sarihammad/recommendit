.PHONY: install download-data features train-embeddings build-index train-ranker train \
        load-features serve test docker-up docker-down evaluate clean lint format

PYTHON := python
MODELS_DIR := models
DATA_DIR := data/ml-1m
FEATURES_DIR := data/features

# ------------------------------------------------------------------ #
# Setup                                                               #
# ------------------------------------------------------------------ #

install:
	pip install -r requirements.txt
	@echo "Dependencies installed."

# ------------------------------------------------------------------ #
# Data                                                                #
# ------------------------------------------------------------------ #

download-data:
	$(PYTHON) data/download.py data
	@echo "MovieLens 1M dataset ready at data/ml-1m/"

# ------------------------------------------------------------------ #
# Training Stages                                                     #
# ------------------------------------------------------------------ #

features:
	$(PYTHON) -m src.pipelines.run_pipeline --stage features \
		--data-dir $(DATA_DIR) --features-dir $(FEATURES_DIR)

load-features:
	$(PYTHON) -m src.pipelines.run_pipeline --stage load_features \
		--data-dir $(DATA_DIR) --features-dir $(FEATURES_DIR)

train-embeddings:
	$(PYTHON) -m src.pipelines.run_pipeline --stage embeddings \
		--data-dir $(DATA_DIR) --models-dir $(MODELS_DIR)

build-index:
	$(PYTHON) -m src.pipelines.run_pipeline --stage index \
		--data-dir $(DATA_DIR) --models-dir $(MODELS_DIR)

train-ranker:
	$(PYTHON) -m src.pipelines.run_pipeline --stage ranker \
		--data-dir $(DATA_DIR) --models-dir $(MODELS_DIR) --features-dir $(FEATURES_DIR)

train: download-data features train-embeddings build-index train-ranker load-features
	@echo "Full training pipeline complete."

# ------------------------------------------------------------------ #
# Evaluation                                                          #
# ------------------------------------------------------------------ #

evaluate:
	$(PYTHON) -m src.pipelines.run_pipeline --stage evaluate \
		--data-dir $(DATA_DIR) --models-dir $(MODELS_DIR)

# ------------------------------------------------------------------ #
# Serving                                                             #
# ------------------------------------------------------------------ #

serve:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --reload

serve-prod:
	uvicorn src.serving.app:app --host 0.0.0.0 --port 8000 --workers 2

# ------------------------------------------------------------------ #
# Testing                                                             #
# ------------------------------------------------------------------ #

test:
	pytest tests/ -v --tb=short

test-features:
	pytest tests/test_features.py -v --tb=short

test-models:
	pytest tests/test_models.py -v --tb=short

test-api:
	pytest tests/test_api.py -v --tb=short

test-coverage:
	pytest tests/ --cov=src --cov-report=html --cov-report=term-missing -v

# ------------------------------------------------------------------ #
# Docker                                                              #
# ------------------------------------------------------------------ #

docker-up:
	docker compose up -d
	@echo "Services started:"
	@echo "  API:        http://localhost:8000"
	@echo "  Prometheus: http://localhost:9090"
	@echo "  Grafana:    http://localhost:3000 (admin/admin)"

docker-down:
	docker compose down

docker-build:
	docker compose build --no-cache

docker-logs:
	docker compose logs -f api

docker-restart:
	docker compose restart api

# ------------------------------------------------------------------ #
# Code Quality                                                        #
# ------------------------------------------------------------------ #

lint:
	@which flake8 > /dev/null 2>&1 && flake8 src/ tests/ --max-line-length=120 --ignore=E501,W503 || echo "flake8 not installed"

format:
	@which black > /dev/null 2>&1 && black src/ tests/ --line-length=100 || echo "black not installed"

type-check:
	@which mypy > /dev/null 2>&1 && mypy src/ --ignore-missing-imports || echo "mypy not installed"

# ------------------------------------------------------------------ #
# Utilities                                                           #
# ------------------------------------------------------------------ #

clean:
	find . -type f -name "*.pyc" -delete
	find . -type d -name "__pycache__" -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name ".pytest_cache" -exec rm -rf {} + 2>/dev/null || true
	find . -name "*.egg-info" -exec rm -rf {} + 2>/dev/null || true

clean-models:
	rm -rf $(MODELS_DIR)/*.pt $(MODELS_DIR)/*.index $(MODELS_DIR)/*.lgbm $(MODELS_DIR)/*.pkl
	@echo "Model artifacts cleaned."

clean-data:
	rm -rf $(FEATURES_DIR)
	@echo "Feature cache cleaned."

pipeline-all:
	$(PYTHON) -m src.pipelines.run_pipeline --stage all

help:
	@echo "RecommendIt Makefile targets:"
	@echo ""
	@echo "  Setup:"
	@echo "    install          Install Python dependencies"
	@echo ""
	@echo "  Data:"
	@echo "    download-data    Download MovieLens 1M dataset"
	@echo ""
	@echo "  Training:"
	@echo "    features         Build user/item features"
	@echo "    load-features    Load features into Redis"
	@echo "    train-embeddings Train two-tower model"
	@echo "    build-index      Build FAISS ANN index"
	@echo "    train-ranker     Train LightGBM ranker"
	@echo "    train            Run complete training pipeline"
	@echo ""
	@echo "  Evaluation:"
	@echo "    evaluate         Run offline evaluation"
	@echo ""
	@echo "  Serving:"
	@echo "    serve            Start API with hot reload (dev)"
	@echo "    serve-prod       Start API (production)"
	@echo ""
	@echo "  Testing:"
	@echo "    test             Run all tests"
	@echo "    test-coverage    Run tests with coverage report"
	@echo ""
	@echo "  Docker:"
	@echo "    docker-up        Start all Docker services"
	@echo "    docker-down      Stop all Docker services"
	@echo "    docker-build     Rebuild Docker images"
