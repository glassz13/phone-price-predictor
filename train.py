# =============================================================
# PHONE PRICE PREDICTION — FINAL ML PIPELINE
# Target    : normalized_used_price
# Benchmark : normalized_new_price (end only)
# Models    : Linear Regression → Random Forest → XGBoost
# =============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor

plt.style.use("seaborn-v0_8-whitegrid")
SEED = 42

# =============================================================
# SECTION 1: LOAD & NULL HANDLING
# =============================================================
print("=" * 60)
print("SECTION 1: LOAD & NULL HANDLING")
print("=" * 60)

df = pd.read_csv("phonedata.csv")
print(f"Shape: {df.shape}")
print(f"\nNulls before:\n{df.isnull().sum()}")

df["rear_camera_mp"]  = df["rear_camera_mp"].fillna(df["rear_camera_mp"].median())
df["front_camera_mp"] = df["front_camera_mp"].fillna(df["front_camera_mp"].median())
df["internal_memory"] = df["internal_memory"].fillna(df["internal_memory"].median())
df["ram"]             = df["ram"].fillna(df["ram"].median())
df["battery"]         = df["battery"].fillna(df["battery"].median())
df["weight"]          = df["weight"].fillna(df["weight"].median())

print(f"\nNulls after : {df.isnull().sum().sum()} — clean!")
print(f"Duplicates  : {df.duplicated().sum()}")

# =============================================================
# SECTION 2: FEATURE ENGINEERING
# =============================================================
print("\n" + "=" * 60)
print("SECTION 2: FEATURE ENGINEERING")
print("=" * 60)

df_model = df.copy()

# 1. Total camera MP (front + rear)
df_model["camera_total"]       = df_model["rear_camera_mp"] + df_model["front_camera_mp"]

# 2. RAM to storage ratio — memory efficiency signal
df_model["ram_storage_ratio"]  = df_model["ram"] / df_model["internal_memory"]

# 3. Battery per gram — energy density vs weight
df_model["battery_per_gram"]   = df_model["battery"] / df_model["weight"]

# 4. Flagship flag — high RAM + high storage tier proxy
df_model["is_flagship"]        = (
    (df_model["ram"] >= 6) & (df_model["internal_memory"] >= 128)
).astype(int)

# 5. Phone age at 2020 baseline
df_model["phone_age"]          = 2020 - df_model["release_year"]

# 6. Connectivity score: 5G=2, 4G=1, neither=0
df_model["connectivity_score"] = (
    (df_model["5g"] == "yes").astype(int) * 2 +
    (df_model["4g"] == "yes").astype(int)
)

# Fix inf from division before anything else
df_model["ram_storage_ratio"] = df_model["ram_storage_ratio"].replace([np.inf, -np.inf], np.nan)
df_model["battery_per_gram"]  = df_model["battery_per_gram"].replace([np.inf, -np.inf], np.nan)
df_model.fillna(df_model.median(numeric_only=True), inplace=True)

print(f"NaNs after engineering : {df_model.isnull().sum().sum()} — clean!")

# Encode binary columns
df_model["4g"] = (df_model["4g"] == "yes").astype(int)
df_model["5g"] = (df_model["5g"] == "yes").astype(int)

# Label encode brand and OS
le_brand = LabelEncoder()
le_os    = LabelEncoder()
df_model["device_brand_enc"] = le_brand.fit_transform(df_model["device_brand"])
df_model["os_enc"]           = le_os.fit_transform(df_model["os"])
df_model.drop(columns=["device_brand", "os"], inplace=True)

print(f"Final shape : {df_model.shape}")

# =============================================================
# SECTION 3: TRAIN / TEST SPLIT
# =============================================================
print("\n" + "=" * 60)
print("SECTION 3: TRAIN / TEST SPLIT")
print("=" * 60)

TARGET    = "normalized_used_price"
BENCHMARK = "normalized_new_price"

benchmark_series = df_model[BENCHMARK].copy().reset_index(drop=True)
X = df_model.drop(columns=[TARGET, BENCHMARK]).reset_index(drop=True)
y = df_model[TARGET].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=SEED
)
bench_test = benchmark_series[y_test.index]

print(f"Train : {X_train.shape[0]} | Test : {X_test.shape[0]} | Features : {X.shape[1]}")

scaler     = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# =============================================================
# EVALUATION HELPER
# =============================================================
results = []

def evaluate(name, model, X_tr, y_tr, X_te, y_te):
    model.fit(X_tr, y_tr)
    preds = model.predict(X_te)
    rmse  = np.sqrt(mean_squared_error(y_te, preds))
    mae   = mean_absolute_error(y_te, preds)
    r2    = r2_score(y_te, preds)
    cv    = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2", n_jobs=-1).mean()
    print(f"\n{'—'*40}")
    print(f"  MODEL : {name}")
    print(f"  RMSE  : {rmse:.4f}")
    print(f"  MAE   : {mae:.4f}")
    print(f"  R²    : {r2:.4f}")
    print(f"  CV R² : {cv:.4f}")
    return {"model": name, "RMSE": rmse, "MAE": mae, "R2": r2, "CV_R2": cv, "preds": preds}

# =============================================================
# SECTION 4: BASELINE — LINEAR REGRESSION
# =============================================================
print("\n" + "=" * 60)
print("SECTION 4: BASELINE — LINEAR REGRESSION")
print("=" * 60)

r1 = evaluate("Linear Regression", LinearRegression(),
              X_train_sc, y_train, X_test_sc, y_test)
results.append({k: v for k, v in r1.items() if k != "preds"})

# =============================================================
# SECTION 5: ENSEMBLE — RANDOM FOREST
# =============================================================
print("\n" + "=" * 60)
print("SECTION 5: ENSEMBLE — RANDOM FOREST")
print("=" * 60)

rf = RandomForestRegressor(n_estimators=200, random_state=SEED, n_jobs=-1)
r2 = evaluate("Random Forest", rf, X_train, y_train, X_test, y_test)
results.append({k: v for k, v in r2.items() if k != "preds"})

# =============================================================
# SECTION 6: BOOSTING — XGBOOST (default)
# =============================================================
print("\n" + "=" * 60)
print("SECTION 6: BOOSTING — XGBOOST (default)")
print("=" * 60)

r3 = evaluate("XGBoost (default)", XGBRegressor(random_state=SEED, verbosity=0),
              X_train, y_train, X_test, y_test)
results.append({k: v for k, v in r3.items() if k != "preds"})

# =============================================================
# SECTION 7: BOOSTING — XGBOOST (tuned)
# =============================================================
print("\n" + "=" * 60)
print("SECTION 7: BOOSTING — XGBOOST (GridSearchCV)")
print("=" * 60)

param_grid = {
    "n_estimators":     [100, 200, 300],
    "max_depth":        [3, 5, 7],
    "learning_rate":    [0.05, 0.1, 0.2],
    "subsample":        [0.8, 1.0],
    "colsample_bytree": [0.8, 1.0],
}

kf       = KFold(n_splits=5, shuffle=True, random_state=SEED)
xgb_grid = GridSearchCV(
    XGBRegressor(random_state=SEED, verbosity=0),
    param_grid, cv=kf, scoring="r2", n_jobs=-1, verbose=1
)
xgb_grid.fit(X_train, y_train)

print(f"\nBest Params : {xgb_grid.best_params_}")
print(f"Best CV R²  : {xgb_grid.best_score_:.4f}")

best_xgb  = xgb_grid.best_estimator_
r4        = evaluate("XGBoost (tuned)", best_xgb, X_train, y_train, X_test, y_test)
results.append({k: v for k, v in r4.items() if k != "preds"})
xgb_preds = r4["preds"]

# =============================================================
# SECTION 8: BENCHMARK CHECK
# =============================================================
print("\n" + "=" * 60)
print("SECTION 8: BENCHMARK — OUR MODEL vs normalized_new_price")
print("=" * 60)

bench_rmse = np.sqrt(mean_squared_error(y_test, bench_test))
bench_r2   = r2_score(y_test, bench_test)
our_rmse   = np.sqrt(mean_squared_error(y_test, xgb_preds))
our_r2     = r2_score(y_test, xgb_preds)

print(f"\n  normalized_new_price (naive)  → RMSE: {bench_rmse:.4f} | R²: {bench_r2:.4f}")
print(f"  XGBoost (tuned)               → RMSE: {our_rmse:.4f}  | R²: {our_r2:.4f}")
print(f"\n  Our model {'✅ BEATS' if our_r2 > bench_r2 else '⚠️ is below'} the benchmark!")

# =============================================================
# SECTION 9: FINAL PLOTS
# =============================================================
print("\n" + "=" * 60)
print("SECTION 9: FINAL PLOTS")
print("=" * 60)

results_df = pd.DataFrame(results)

# --- PLOT 1: Model Comparison ---
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle("Model Progression — Baseline → Boosting", fontsize=15, fontweight="bold")
for i, (metric, color) in enumerate(zip(
    ["RMSE", "MAE", "R2"],
    ["#C44E52", "#DD8452", "#4C72B0"]
)):
    bars = axes[i].bar(results_df["model"], results_df[metric],
                       color=color, edgecolor="white", width=0.6)
    axes[i].set_title(metric, fontsize=13)
    axes[i].tick_params(axis="x", rotation=35)
    axes[i].set_ylabel(metric)
    for bar in bars:
        axes[i].text(bar.get_x() + bar.get_width() / 2,
                     bar.get_height() + 0.002,
                     f"{bar.get_height():.3f}",
                     ha="center", va="bottom", fontsize=8)
plt.tight_layout()
plt.savefig("plot1_model_comparison.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plot1_model_comparison.png")

# --- PLOT 2: XGBoost Feature Importance ---
feat_imp = pd.Series(best_xgb.feature_importances_, index=X.columns).sort_values(ascending=False)
fig, ax  = plt.subplots(figsize=(10, 6))
feat_imp.head(15).plot(kind="bar", ax=ax, color="#C44E52", edgecolor="white")
ax.set_title("XGBoost (Tuned) — Feature Importance", fontsize=13, fontweight="bold")
ax.set_ylabel("Importance Score")
ax.tick_params(axis="x", rotation=45)
plt.tight_layout()
plt.savefig("plot2_feature_importance.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plot2_feature_importance.png")

# --- PLOT 3: Actual vs Predicted ---
fig, ax = plt.subplots(figsize=(8, 6))
ax.scatter(y_test, xgb_preds, alpha=0.4, color="#4C72B0", s=20, label="Predictions")
ax.plot([y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "r--", lw=2, label="Perfect Fit")
ax.set_xlabel("Actual Price", fontsize=12)
ax.set_ylabel("Predicted Price", fontsize=12)
ax.set_title("XGBoost Tuned — Actual vs Predicted", fontsize=13, fontweight="bold")
ax.legend()
plt.tight_layout()
plt.savefig("plot3_actual_vs_predicted.png", dpi=150, bbox_inches="tight")
plt.show()
print("Saved: plot3_actual_vs_predicted.png")

# =============================================================
# FINAL SUMMARY
# =============================================================
print("\n" + "=" * 60)
print("FINAL RESULTS SUMMARY")
print("=" * 60)
print(f"\n{results_df[['model','RMSE','MAE','R2','CV_R2']].to_string(index=False)}")
print(f"\nBenchmark (normalized_new_price) → R²: {bench_r2:.4f} | RMSE: {bench_rmse:.4f}")
print(f"Our Best  (XGBoost tuned)        → R²: {our_r2:.4f}  | RMSE: {our_rmse:.4f}")
print(f"\n{'✅ PIPELINE COMPLETE':^60}")

# =============================================================
# SECTION 10: SAVE MODEL & ARTIFACTS
# =============================================================
import pickle

print("\n" + "=" * 60)
print("SECTION 10: SAVING MODEL & ARTIFACTS")
print("=" * 60)

artifacts = {
    "model":         best_xgb,
    "le_brand":      le_brand,
    "le_os":         le_os,
    "scaler":        scaler,
    "known_brands":  sorted(df["device_brand"].unique().tolist()),
    "known_os":      sorted(df["os"].unique().tolist()),
}

with open("model.pkl", "wb") as f:
    pickle.dump(artifacts, f)

print("✅ model.pkl saved!")
print(f"   Brands : {len(artifacts['known_brands'])}")
print(f"   OS     : {len(artifacts['known_os'])}")