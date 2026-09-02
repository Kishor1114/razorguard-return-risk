# RazorGuard — Return Risk Prediction System

RazorGuard is an explainable return-risk prediction system for e-commerce orders.

The system predicts the probability that an order will be returned, assigns a risk level, identifies the strongest model drivers using SHAP, and generates a human-readable explanation using an LLM.

The pipeline is designed with input validation, leakage-safe feature construction, structured evidence validation, and automated tests to make the prediction and explanation process reliable and auditable.

---

## Overview

E-commerce businesses need to identify potentially high-risk orders early enough to make informed operational decisions.

RazorGuard addresses this problem through an end-to-end machine learning pipeline:

```text
Order Input
    |
    v
Input Validation
    |
    v
CatBoost Return-Risk Model
    |
    v
Risk Probability
    |
    v
SHAP Feature Attribution
    |
    v
Structured Evidence
    |
    +----------------------+
    |                      |
    v                      v
Deterministic          LLM Explanation
Explanation                 |
                            v
                    LLM Explanation
                       Validation
                            |
                            v
                     Final Prediction
```

The LLM is used only as an explanation layer. It does not calculate the prediction, modify the probability, or determine the risk level.

---

## Key Features

### 1. Leakage-Safe Feature Construction

The synthetic dataset is generated chronologically using historical information available before each order.

For every order, historical features are calculated from running customer and category history before the current order's outcome is added.

This prevents:

- Using the current order's outcome as a feature
- Using future orders to construct historical features
- Target leakage through post-order information

The leakage-safe design is verified through independent recomputation tests.

---

### 2. CatBoost Return-Risk Model

The current frozen baseline model is a CatBoost classifier trained to predict order return risk.

The model uses 15 features:

```text
return_rate_90d
category_return_rate
is_cod
order_value
discount_pct
delivery_distance_km
previous_returns
customer_age_days
previous_orders
product_rating
orders_last_30d
region_code
browser_family
device_age_days
session_hour
```

Categorical features:

```text
region_code
browser_family
```

The trained model artifacts are stored under:

```text
models/
├── razorguard_catboost_baseline.cbm
└── razorguard_catboost_baseline.pkl
```

---

## Model Performance

The frozen baseline model currently records:

| Metric | Score |
|---|---:|
| ROC-AUC | 0.6733 |
| PR-AUC | 0.3341 |

These metrics are preserved as part of the structured prediction metadata.

---

## Risk Classification

The predicted probability is converted into a risk level using fixed thresholds:

| Probability | Risk Level |
|---:|---|
| `< 0.20` | LOW |
| `0.20 – < 0.50` | MEDIUM |
| `>= 0.50` | HIGH |

Example:

```text
Probability: 0.0846
Risk Level: LOW
```

The model probability remains the source of truth for risk classification.

---

## Explainability with SHAP

RazorGuard uses SHAP to identify the features contributing most strongly to an individual prediction.

For each selected feature, the system records:

```text
feature
value
shap_value
absolute_shap
direction
```

The direction is derived directly from the SHAP value:

```text
SHAP > 0  -> increases_risk
SHAP < 0  -> decreases_risk
SHAP = 0  -> neutral
```

The strongest drivers are sorted by absolute SHAP magnitude and the top five are retained as structured evidence.

Example:

```json
{
  "feature": "orders_last_30d",
  "value": 4,
  "shap_value": -0.548,
  "absolute_shap": 0.548,
  "direction": "decreases_risk"
}
```

This structured evidence is used as the source of truth for downstream explanations.

---

## Deterministic Explanation

Before using an LLM, RazorGuard generates a deterministic explanation directly from the model output and SHAP evidence.

Example:

```text
Predicted return risk is 8.5% (LOW).
orders_last_30d = 4 decreases risk;
is_cod = True increases risk;
order_value = 1121.56 decreases risk.
```

This provides an auditable explanation that does not depend on generative model behavior.

---

## LLM Explanation Layer

The system can generate a natural-language explanation using an LLM.

The LLM receives only:

- Risk probability
- Risk level
- Top SHAP drivers
- Feature values
- SHAP values
- SHAP directions

The LLM does not receive the complete raw order record.

The explanation prompt explicitly requires the model to:

- Preserve the supplied SHAP directions
- Mention the supplied top drivers
- Use the supplied feature names and values
- Avoid inventing customer or product information
- Avoid unsupported features
- Avoid claiming causation
- Avoid recommending actions
- Stay within the specified explanation length

This design keeps the LLM focused on explanation rather than prediction.

---

## Explanation Validation

Generated explanations are validated against the structured prediction evidence before being returned as a successful result.

The validator checks that:

- The explanation exists and is non-empty
- Structured prediction evidence is valid
- Required features are respected
- Unsupported features are not introduced
- The supplied prediction information is preserved
- SHAP directions are not contradicted

If validation fails, the pipeline returns a `FAILED` status instead of silently accepting potentially unreliable output.

---

## Input Validation

Every order is validated before it reaches the machine learning model.

The validator checks:

### Required Features

All 15 model features must be present.

### Numeric Features

Numeric fields must contain valid numeric values and cannot be missing.

### Boolean Features

`is_cod` must contain a boolean value.

### Example Validation Failures

```text
Feature 'order_value' must be numeric.
```

```text
Missing required features: ['order_value']
```

```text
Feature 'order_value' cannot be missing.
```

```text
Feature 'is_cod' must be boolean.
```

Multiple validation errors are returned together where applicable.

---

## End-to-End Pipeline

The main pipeline is implemented through:

```python
run_end_to_end(order_row)
```

The function performs:

1. Input validation
2. Input preparation
3. CatBoost prediction
4. Risk classification
5. SHAP calculation
6. Structured evidence generation
7. Structured evidence validation
8. Deterministic explanation generation
9. LLM explanation generation
10. LLM explanation validation
11. Final result construction

A successful response follows this general structure:

```json
{
  "status": "SUCCESS",
  "prediction": {
    "probability": 0.0846,
    "risk_level": "LOW"
  },
  "evidence": [],
  "deterministic_explanation": "...",
  "llm_explanation": "...",
  "validation": {
    "structured": {
      "valid": true,
      "errors": []
    },
    "llm": {
      "valid": true,
      "errors": []
    }
  }
}
```

Invalid inputs are rejected before model inference:

```json
{
  "status": "FAILED",
  "errors": [
    "Feature 'order_value' must be numeric."
  ]
}
```

---

## Dataset

The project uses a reproducible synthetic e-commerce dataset containing:

- 10,000 orders
- January–August 2026 time period
- Approximately 18% observed return rate
- Chronological feature construction
- Customer and product/category historical signals

The dataset is generated with a fixed seed for reproducibility.

Observed results from the dataset generation:

| Split | Orders | Return Rate |
|---|---:|---:|
| Train | 4,929 | 17.9% |
| Validation | 1,294 | 17.0% |
| Test | 3,777 | 18.6% |

The strongest empirical signals include:

- `category_return_rate`
- `order_value`
- `return_rate_90d`
- `is_cod`

`region_code` and `browser_family` are intentionally non-predictive or weak signals in the generated data.

---

## Project Structure

```text
razorguard/
|
├── data/
│   └── razorguard_dataset.csv
|
├── models/
│   ├── razorguard_catboost_baseline.cbm
│   └── razorguard_catboost_baseline.pkl
|
├── notebooks/
│   ├── 01_data_validation.ipynb
│   └── 02_baseline_models.ipynb
|
├── src/
│   ├── data_generator.py
│   ├── pipeline.py
│   └── validator.py
|
├── tests/
│   ├── test_leakage.py
│   ├── test_pipeline.py
│   └── test_validator.py
|
├── .env
├── .gitignore
├── README.md
└── requirements.txt
```

> `.env` and virtual-environment files should remain excluded from version control.

---

## Installation

Create and activate a virtual environment.

### Windows PowerShell

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Install dependencies:

```powershell
pip install -r requirements.txt
```

The project dependencies include:

```text
pandas
numpy
scikit-learn
catboost
shap
groq
python-dotenv
pytest
```

---

## Environment Configuration

The LLM explanation layer uses an API key loaded through environment variables.

Create a `.env` file:

```text
GROQ_API_KEY=your_api_key_here
```

Never commit the `.env` file to version control.

---

## Running the Dataset Generator

To regenerate the synthetic dataset:

```powershell
python src/data_generator.py
```

This generates:

```text
data/razorguard_dataset.csv
```

---

## Running Tests

Run the complete test suite:

```powershell
python -m pytest -q --disable-warnings
```

Current test status:

```text
20 passed
```

The test suite covers:

- Leakage prevention
- Input validation
- Missing features
- Invalid numeric values
- Missing numeric values
- Invalid boolean values
- Multiple invalid fields
- Structured prediction validation
- SHAP evidence validation
- LLM explanation validation
- End-to-end pipeline behavior

---

## Running Individual Test Suites

Leakage tests:

```powershell
python -m pytest tests\test_leakage.py -v
```

Pipeline tests:

```powershell
python -m pytest tests\test_pipeline.py -v
```

Validator tests:

```powershell
python -m pytest tests\test_validator.py -v
```

For concise output:

```powershell
python -m pytest -q --disable-warnings
```

---

## Reproducibility

The project is designed to make the core data-generation and prediction workflow reproducible.

Key reproducibility decisions include:

- Fixed random seed for synthetic data generation
- Frozen baseline model artifacts
- Explicit model feature ordering
- Explicit categorical feature definitions
- Deterministic risk thresholds
- Structured SHAP evidence
- Automated validation tests

---

## Design Principles

### Model Output Is the Source of Truth

The LLM does not determine the risk score or risk level.

### Structured Evidence Before Generation

The LLM receives validated, structured evidence rather than unrestricted raw data.

### Explainability Must Be Auditable

SHAP values and directions are retained as structured evidence.

### Fail Closed

Invalid input or invalid explanation output results in a `FAILED` pipeline response rather than silently returning potentially unreliable results.

### Leakage Safety by Construction

Historical features are generated from information available before the current order.

---

## Current Status

RazorGuard currently provides a working baseline return-risk prediction and explanation pipeline with automated validation.

Implemented components:

- [x] Synthetic e-commerce dataset generation
- [x] Leakage-safe historical feature construction
- [x] Leakage validation tests
- [x] CatBoost baseline model
- [x] Frozen model artifacts
- [x] Input schema and type validation
- [x] Risk probability prediction
- [x] Risk-level classification
- [x] SHAP feature attribution
- [x] Structured evidence generation
- [x] Deterministic explanations
- [x] LLM-generated explanations
- [x] LLM explanation validation
- [x] End-to-end pipeline
- [x] Automated test suite

---

## Notes

The current model is a baseline model and is not intended to represent a production-calibrated return-risk system.

The synthetic dataset is designed for development and evaluation of the RazorGuard pipeline. Model performance should not be interpreted as production business performance.

---

## Project

**RazorGuard**

An explainable e-commerce return-risk prediction system built around leakage-safe features, machine learning, SHAP-based evidence, and validated LLM explanations.