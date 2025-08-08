# ApollodB helpers
PY?=python3
PIP?=$(PY) -m pip
UVICORN?=$(PY) -m uvicorn
APP=backend.server:app
PORT?=8000
HOST?=127.0.0.1

.PHONY: venv install dev run-backend run-frontend fmt lint test docker-build docker-run gcloud-run

venv:
	$(PY) -m venv .venv
	. .venv/bin/activate; $(PIP) install --upgrade pip

install:
	. .venv/bin/activate; $(PIP) install -r requirements.txt

dev: run-backend run-frontend

run-backend:
	. .venv/bin/activate; $(UVICORN) $(APP) --host $(HOST) --port $(PORT) --reload

run-frontend:
	$(PY) -m http.server 8080 -d web

fmt:
	. .venv/bin/activate; python -m pip install ruff black
	. .venv/bin/activate; ruff check --select I --fix . || true
	. .venv/bin/activate; black .

lint:
	. .venv/bin/activate; ruff check .

# basic placeholder for tests
test:
	@echo "No test suite yet"

# Docker
IMAGE?=gcr.io/$(PROJECT_ID)/apollodb

docker-build:
	docker build -t $(IMAGE) .

docker-run:
	docker run --rm -p 8080:8080 $(IMAGE)

# Cloud Run deploy (requires gcloud auth and PROJECT_ID)
REGION?=us-central1
SERVICE?=apollodb

gcloud-run:
	gcloud run deploy $(SERVICE) --source . --region $(REGION) --project $(PROJECT_ID) --platform managed --allow-unauthenticated --port 8080
