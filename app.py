# app.py — Phone Used Price Predictor
# Run: streamlit run app.py

import streamlit as st
import pandas as pd
import numpy as np
import pickle

# =============================================================
# PAGE CONFIG
# =============================================================
st.set_page_config(
    page_title="Phone Price Predictor",
    page_icon="📱",
    layout="centered"
)

# =============================================================
# LOAD MODEL & ARTIFACTS
# =============================================================
@st.cache_resource
def load_artifacts():
    with open("model.pkl", "rb") as f:
        artifacts = pickle.load(f)
    return artifacts

artifacts    = load_artifacts()
model        = artifacts["model"]
le_brand     = artifacts["le_brand"]
le_os        = artifacts["le_os"]
known_brands = artifacts["known_brands"]
known_os     = artifacts["known_os"]

# =============================================================
# PRICE CONVERSION — log scale → dollars (2015–2020 era)
# =============================================================
def normalized_to_dollars(val):
    # Observed range: ~2.5 (cheapest) to ~6.0 (most expensive)
    # Mapped linearly to realistic used phone prices 2015-2020
    min_norm, max_norm = 2.5, 6.0
    min_usd,  max_usd  = 40,  900
    dollar = ((val - min_norm) / (max_norm - min_norm)) * (max_usd - min_usd) + min_usd
    return max(40, round(dollar, -1))  # round to nearest $10, floor at $40

# =============================================================
# HEADER
# =============================================================
st.title("📱 Phone Used Price Predictor")
st.warning(
    "⚠️ **Note:** This model is trained on a dataset from 2015–2020. "
    "Predictions reflect phone market pricing from that era and may not "
    "accurately represent current market values."
)
st.markdown("---")
st.markdown("### Enter Phone Specifications")

# =============================================================
# INPUT FORM
# =============================================================
col1, col2 = st.columns(2)

with col1:
    st.markdown("**📐 Display & Build**")
    screen_size = st.slider("Screen Size (inches)", 4.0, 10.0, 6.1, step=0.1)
    weight      = st.slider("Weight (grams)", 100, 500, 180, step=5)

    st.markdown("**💾 Storage & Memory**")
    internal_memory = st.selectbox("Internal Memory (GB)", [16, 32, 64, 128, 256, 512])
    ram             = st.selectbox("RAM (GB)", [1, 2, 3, 4, 6, 8, 12, 16])

with col2:
    st.markdown("**📷 Camera**")
    rear_camera_mp  = st.slider("Rear Camera (MP)", 2, 108, 12, step=1)
    front_camera_mp = st.slider("Front Camera (MP)", 2, 64, 8, step=1)

    st.markdown("**🔋 Battery & Connectivity**")
    battery = st.slider("Battery (mAh)", 1500, 10000, 4000, step=100)
    fg      = st.radio("4G", ["Yes", "No"], horizontal=True)
    fg5     = st.radio("5G", ["Yes", "No"], horizontal=True)

st.markdown("---")
col3, col4 = st.columns(2)

with col3:
    st.markdown("**📅 Release Info**")
    release_year = st.selectbox("Release Year", list(range(2015, 2021)), index=5)
    days_used    = st.slider("Days Used", 0, 1200, 300, step=10)

with col4:
    st.markdown("**🏷️ Brand & OS**")
    brand = st.selectbox("Brand", sorted(known_brands))
    os    = st.selectbox("Operating System", sorted(known_os))

# =============================================================
# FEATURE ENGINEERING
# =============================================================
def build_features(screen_size, rear_camera_mp, front_camera_mp,
                   internal_memory, ram, battery, weight,
                   release_year, days_used, fg, fg5, brand, os):

    camera_total       = rear_camera_mp + front_camera_mp
    ram_storage_ratio  = ram / internal_memory if internal_memory > 0 else 0
    battery_per_gram   = battery / weight if weight > 0 else 0
    is_flagship        = int(ram >= 6 and internal_memory >= 128)
    phone_age          = 2020 - release_year
    connectivity_score = (int(fg5 == "Yes") * 2) + int(fg == "Yes")
    fg_enc             = 1 if fg  == "Yes" else 0
    fg5_enc            = 1 if fg5 == "Yes" else 0

    try:
        brand_enc = le_brand.transform([brand])[0]
    except ValueError:
        brand_enc = 0
    try:
        os_enc = le_os.transform([os])[0]
    except ValueError:
        os_enc = 0

    features = pd.DataFrame([{
        "screen_size":        screen_size,
        "4g":                 fg_enc,
        "5g":                 fg5_enc,
        "rear_camera_mp":     rear_camera_mp,
        "front_camera_mp":    front_camera_mp,
        "internal_memory":    internal_memory,
        "ram":                ram,
        "battery":            battery,
        "weight":             weight,
        "release_year":       release_year,
        "days_used":          days_used,
        "camera_total":       camera_total,
        "ram_storage_ratio":  ram_storage_ratio,
        "battery_per_gram":   battery_per_gram,
        "is_flagship":        is_flagship,
        "phone_age":          phone_age,
        "connectivity_score": connectivity_score,
        "device_brand_enc":   brand_enc,
        "os_enc":             os_enc,
    }])
    return features

# =============================================================
# PREDICT
# =============================================================
st.markdown("---")

if st.button("🔮 Predict Used Price", use_container_width=True, type="primary"):
    input_df   = build_features(
        screen_size, rear_camera_mp, front_camera_mp,
        internal_memory, ram, battery, weight,
        release_year, days_used, fg, fg5, brand, os
    )

    prediction = model.predict(input_df)[0]
    dollars    = normalized_to_dollars(prediction)

    # Price tier
    if prediction < 3.5:
        tier, tier_color = "🟢 Budget", "green"
        price_range = "$40 – $150"
    elif prediction < 4.5:
        tier, tier_color = "🟡 Mid-Range", "orange"
        price_range = "$150 – $400"
    elif prediction < 5.2:
        tier, tier_color = "🟠 Premium", "orange"
        price_range = "$400 – $700"
    else:
        tier, tier_color = "🔴 Flagship", "red"
        price_range = "$700+"

    # --- OUTPUT ---
    st.markdown("---")
    st.markdown("### 💰 Prediction Result")

    c1, c2, c3 = st.columns(3)
    c1.metric("Estimated Used Price", f"${dollars}")
    c2.metric("Price Range", price_range)
    c3.metric("Segment", tier)

    st.markdown("---")

    # --- WHAT'S DRIVING THIS PRICE ---
    st.markdown("### 🔍 What's Driving This Price")
    st.caption("Based on XGBoost feature importances — how much each spec contributes to used phone pricing.")

    # Fixed importances from trained model (from feature importance chart)
    drivers = {
        "📐 Screen Size":      22,
        "📷 Total Camera MP":  18,
        "💾 Internal Memory":  11,
        "📶 4G Support":        9,
        "🔋 Battery":           6,
        "⭐ Flagship Tier":     5,
        "Other Specs":          29,
    }

    for label, pct in drivers.items():
        st.markdown(f"**{label}** — {pct}%")
        st.progress(pct / 100)

    st.markdown("---")
    st.caption(
        f"Model: XGBoost (tuned) · R² = 0.8186 · "
        f"Trained on {3454} phones from 2015–2020"
    )