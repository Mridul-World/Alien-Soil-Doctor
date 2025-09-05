import streamlit as st
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from io import StringIO
import os

# 🧠 Load model once
@st.cache_resource
def load_model():
    model_path = r"C:\Work_place\isru project v1\models\alien_soil_model_ep10000.keras"
    return tf.keras.models.load_model(model_path)

model = load_model()

# 🔧 Setup label maps
label_map = {"not_habitable": 0, "moderate": 1, "habitat": 2}
reverse_map = {0: "not_habitable", 1: "moderate", 2: "habitat"}

# 💡 Suggestion engine
def suggest_improvements(sample):
    tips = []
    if sample["phh2o"] < 6.0:
        tips.append("🧪 Raise pH: Add alkaline minerals (e.g., lime).")
    elif sample["phh2o"] > 8.0:
        tips.append("🧪 Lower pH: Add sulfur compounds or organic matter.")
    if sample["nitrogen"] < 0.5:
        tips.append("🌱 Boost Nitrogen: Add nitrogen-fixing microbes or ammonium salts.")
    if sample["alien_chemical_index"] > 0.6:
        tips.append("☢️ Reduce alien toxicity: Isolate or chemically neutralize unknown compounds.")
    if sample["radiation_level"] > 0.8:
        tips.append("🛡 Shield Radiation: Add subsurface shielding or build protective domes.")
    if sample["soc"] < 0.3:
        tips.append("🪵 Increase Organic Carbon: Mix composted biomass or carbon-rich regolith.")
    if not tips:
        tips.append("✅ Soil is close to optimal. Minor improvements may suffice.")
    return tips

# 📋 UI
st.title("🧠 Alien Soil Doctor v2")
st.subheader("📤 Upload a planetary soil CSV")
uploaded_file = st.file_uploader("Limit 200MB per file • CSV", type=["csv"])

if uploaded_file:
    # 🧾 Read file
    df = pd.read_csv(uploaded_file)

    # 🚨 Map labels if needed
    if "habitat" in df.columns:
        df["habitat"] = df["habitat"].map(label_map)

    # 🧱 Fix depth
    if "depth" in df.columns and df["depth"].dtype == object:
        df["depth"] = df["depth"].str.extract(r"(\d+)").astype(float)

    # ✅ Required features
    required_columns = [
        "lat", "lon", "depth", "phh2o", "soc", "clay", "sand", "silt", "cec",
        "bdod", "nitrogen", "radiation_level", "alien_chemical_index",
        "reward_score", "confidence"
    ]

    for col in required_columns:
        if col not in df.columns:
            df[col] = 0.0

    X = df[required_columns].values
    y_true = df["habitat"].values if "habitat" in df.columns else None

    # ⚖️ Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 🔮 Predict
    pred_probs = model.predict(X_scaled)
    y_pred = np.argmax(pred_probs, axis=1)
    confidences = np.max(pred_probs, axis=1)

    # 🪄 Build output
    output_df = df.copy()
    output_df["Predicted_Label"] = [reverse_map[p] for p in y_pred]
    output_df["Confidence"] = confidences

    # 🧪 Add suggestions column
    st.subheader("🔬 Top Predictions & Suggestions")
    preview_text = ""
    for i in range(min(5, len(y_pred))):
        label = reverse_map[y_pred[i]]
        confidence = confidences[i]
        verdict = "✅ HABITABLE" if y_pred[i] == 2 else "❌ NOT HABITABLE"
        preview_text += f"\nSample {i+1} → Pred={label} | Confidence={confidence:.2f} → {verdict}"
        if y_pred[i] != 2:
            sample_dict = dict(zip(required_columns, X[i]))
            suggestions = suggest_improvements(sample_dict)
            for tip in suggestions:
                preview_text += f"\n   💡 {tip}"
    st.code(preview_text, language="markdown")

    # 📥 Download link
    csv = output_df.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Results CSV", csv, "alien_soil_predictions.csv", "text/csv")

else:
    st.warning("📂 Please upload a CSV file to begin.")
