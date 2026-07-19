# Validation report

Validation date: 19 July 2026

## Automated tests

- Result: **8 passed**
- Coverage areas: FastAPI health and RAG endpoints, complaint intent/severity classification, multilingual input, hybrid SOP retrieval, document extraction, occupancy/ADR/RevPAR response, and end-to-end workflow routing.

## XGBoost forecasting validation

The included model was trained on 2,400 synthetic chronological hotel-demand records, with the final 20% held out by date.

- Backend: XGBoost
- Training rows: 1,920
- Validation MAE: 0.02334 occupancy points
- Validation RMSE: 0.02862 occupancy points
- Validation R²: 0.91060

These figures describe the included synthetic dataset only and should not be represented as production OYO results.

## Workflow benchmark

- Events requested: 10,000
- Successful events: 10,000
- Local benchmark reliability: 100%
- Concurrency: 50
- Duration: 7.8525 seconds
- Throughput: 1,273.48 events/second
- Mean latency: 0.765 ms
- P95 latency: 0.937 ms

The benchmark used in-memory fallbacks for Kafka, Redis, and database persistence. Production infrastructure performance will differ.

## Infrastructure validation

- Python modules compile successfully.
- Lightweight application tests pass.
- The Docker Compose file is included for PostgreSQL, Redis, Kafka, and Weaviate.
- Docker was not available in the build environment, so the full container stack was not started during packaging.
- Transformer and LayoutLMv3 checkpoints are not bundled. Fine-tuning and inference integration scripts are included.
