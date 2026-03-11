# Phone Used Price Prediction — ML Regression Pipeline

> Regression pipeline to predict normalized used phone prices using hardware specs and engineered features. Progression from Linear Regression to tuned XGBoost with a naive benchmark comparison.

---

## Problem Statement

Predict the **normalized used price** of a smartphone given its hardware specifications, connectivity, and usage history.

A naive baseline exists — `normalized_new_price` strongly correlates with used price. The goal is to beat this baseline using only hardware features, with no price leakage.

---

## Project Structure
```
phone-price-prediction/
│
├── train.py
├── phonedata.csv
├── requirements.txt
└── outputs/
    ├── model_comparison.png
    ├── feature_importance.png
    ├── actual_vs_predicted.png
    └── residuals.png
```

---

## Dataset

| Property | Detail |
|----------|--------|
| Target | `normalized_used_price` |
| Benchmark | `normalized_new_price` (comparison only, never used as feature) |
| Total records | 3,454 phones |
| Era | 2015 – 2020 |
| Source | [add dataset link here] |

**Features:** device brand, OS, screen size, 4G/5G, rear/front camera MP, internal memory, RAM, battery, weight, release year, days used

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| `camera_total` | Rear + front camera MP combined |
| `ram_storage_ratio` | RAM / internal memory |
| `battery_per_gram` | Battery / weight |
| `is_flagship` | 1 if RAM >= 6GB and storage >= 128GB |
| `phone_age` | 2020 - release year |
| `connectivity_score` | 5G=2, 4G=1, neither=0 |

Final feature count after engineering and encoding: **19 features**

---

## Workflow
```
Load & Null Handling → Feature Engineering → Train/Test Split (80/20)
    → Linear Regression → Random Forest
    → XGBoost (default) → XGBoost (GridSearchCV)
    → Benchmark Comparison
```

---

## Results

| Model | RMSE | MAE | R² | CV R² |
|-------|------|-----|----|-------|
| Linear Regression | 0.2917 | 0.2168 | 0.7378 | 0.7429 |
| Random Forest | 0.2455 | 0.1936 | 0.8143 | 0.7969 |
| XGBoost (default) | 0.2548 | 0.2007 | 0.8000 | 0.7729 |
| **XGBoost (tuned)** | **0.2426** | **0.1884** | **0.8186** | **0.8092** |

### Benchmark Comparison

| | RMSE | R² |
|--|------|-----|
| normalized_new_price (naive) | 0.9452 | -1.7528 |
| XGBoost tuned (our model) | 0.2426 | 0.8186 |

---

## Model Analysis

**Model Progression**

Linear Regression establishes a reasonable baseline at R² = 0.74, confirming that phone pricing has real linear structure. Random Forest jumps to 0.814 — a significant gain — because it captures non-linear interactions between specs like RAM, storage, and connectivity that linear models miss.

XGBoost default slightly underperforms Random Forest (0.800 vs 0.814) which is common when boosting hyperparameters are left at defaults. After GridSearchCV tuning, XGBoost tuned reaches 0.819 — the best result overall, and with a CV R² of 0.809 confirming it generalizes well rather than overfitting.

**Why XGBoost Wins**

Boosting builds trees sequentially, each one correcting the errors of the previous. For tabular data with mixed feature types like this dataset — numerical specs alongside encoded categoricals — this sequential error correction consistently outperforms Random Forest's parallel ensemble approach. The tuning also found shallow trees (max_depth=3) work best, suggesting the pricing signal is relatively smooth rather than highly complex.

**The Benchmark Result**

The naive baseline using `normalized_new_price` as a predictor scores R² = -1.75 — worse than predicting the mean price for every phone. This is because used price and new price are on different scales and the relationship is non-linear. Our model achieves R² = 0.82 using hardware specs alone, proving the engineered features carry genuine predictive signal.

**Feature Importance Findings**

| Feature | Importance |
|---------|------------|
| screen_size | 0.220 |
| camera_total | 0.175 |
| internal_memory | 0.111 |
| 4g | 0.095 |
| rear_camera_mp | 0.090 |

Screen size dominates — in the 2015–2020 era, display size was the primary consumer differentiator. The engineered `camera_total` feature (front + rear MP combined) ranks second above individual camera features, validating that combining them captures more signal than keeping them separate. 4G connectivity ranking highly reflects the transition period where 4G was still a meaningful differentiator rather than standard.

**Residual Analysis**

Residuals are centered around zero with no obvious systematic pattern, confirming the model has no major bias. The slight spread at lower price ranges suggests budget phones have more pricing variability that is harder to capture from specs alone.

---

## Visualizations

**Model Progression**
![model_comparison](outputs/model_comparison.png)

**XGBoost Feature Importance**
![feature_importance](outputs/feature_importance.png)

**Actual vs Predicted**
![actual_vs_predicted](outputs/actual_vs_predicted.png)

**Residual Analysis**
![residuals](outputs/residuals.png)

---

## Tech Stack

- Python 3.x
- pandas, numpy
- scikit-learn
- xgboost
- matplotlib, seaborn

---

## How to Run
```bash
pip install -r requirements.txt
python train.py
```

---

## requirements.txt
```
pandas
numpy
scikit-learn
xgboost
matplotlib
seaborn
```
