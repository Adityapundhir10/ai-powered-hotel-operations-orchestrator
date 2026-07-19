# Project integrity and interview guidance

## What this repository is

A clean-room portfolio reconstruction that implements the same categories of models, services, and infrastructure described in Aditya Pundhir's resume. It is safe to demonstrate because its data and SOPs are synthetic.

## What this repository is not

- It is not an export of an OYO production repository.
- It does not prove the exact production scale or business-impact percentages stated elsewhere.
- It does not contain private booking records, hotel identifiers, customer information, credentials, or internal documents.

## How to describe it accurately

> During my OYO internship, I worked on hotel-demand forecasting and operations automation. To demonstrate the architecture publicly without exposing confidential code or data, I recreated the system as a clean-room portfolio project using synthetic data. The repository shows the forecasting, pricing, classification, retrieval, orchestration, and event-driven components end to end.

## Before quoting metrics

Run:

```bash
python scripts/generate_synthetic_data.py
python scripts/train_forecaster.py
python scripts/evaluate_models.py
python scripts/benchmark_workflows.py --events 10000 --concurrency 50
```

Use the JSON files written under `artifacts/` as the source for public benchmark numbers.
