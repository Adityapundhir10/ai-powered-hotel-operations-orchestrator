.PHONY: install install-full data train run test benchmark docker

install:
	python -m pip install -r requirements.txt

install-full:
	python -m pip install -r requirements-full.txt

data:
	python scripts/generate_synthetic_data.py

train:
	python scripts/train_forecaster.py

run:
	uvicorn app.main:app --reload

test:
	pytest -q

benchmark:
	python scripts/benchmark_workflows.py --events 10000 --concurrency 50

docker:
	docker compose up --build
