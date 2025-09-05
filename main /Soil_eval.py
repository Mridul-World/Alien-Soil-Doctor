import os
import argparse
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf

# ---------------------------
# 🌍 Generalized Soil Evaluation Script
# ---------------------------

# 📌 Parse input arguments
parser = argparse.ArgumentParser(description="Alien Soil Doctor - Evaluate planetary soil dataset")
parser.add_argument("--data", type=str, required=True, help="Path to input CSV dataset")
parser.add_argument("--model", type=str, required=True, help="Path to trained Keras model (.keras)")
parser.add_argument("--output", type=str, default="alien_soil_predictions.csv", help="Path to save predictions CSV")
args = parser.parse_args()

# 📂 Load dataset
print(f"📦 Loading dataset: {args.data}")
df = pd.read_csv(args.data)

# 🗺️ Label mapping
label_map = {"not_habitable": 0, "moderate": 1, "habitat": 2}
reverse_map = {v: k for k, v in label_map.items()}
if "habitat" in df.columns:
    df["habitat"] = df["habitat"].map(label_map)

# 🧮 Fix depth format (if string like "30cm")
if "depth" in df.columns and df["depth"].dtype == object:
    df["depth"] = df["depth"].str.extract(r"(\d+)").astype(float)

# 🧱 Required model columns
required_columns = [
    "lat", "lon", "depth", "phh2o", "soc", "clay", "sand", "silt", "cec",
    "bdod", "nitrogen", "radiation_level", "alien_chemical_index",
    "reward_score", "confidence"
]

for col in required_columns:
    if col not in df.columns:
        print(f"⚠️ Missing column: {col}, filling with 0.0")
        df[col] = 0.0

# 🎯 Features and labels
X = df[required_columns].values
y_true = df["habitat"].values if "habitat" in df.columns else None

# ⚖️ Standardize inputs
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🧠 Load trained model
print(f"📥 Loading model: {args.model}")
model = tf.keras.models.load_model(args.model)

# 🔮 Predict
print("🧠 Making predictions...")
pred_probs = model.predict(X_scaled)
y_pred = np.argmax(pred_probs, axis=1)
confidences = np.max(pred_probs, axis=1)

# 🎯 Evaluate predictions (if ground truth available)
if y_true is not None:
    print("\n📋 Classification Report (Full Dataset):")
    print(classification_report(y_true, y_pred, digits=4))

    # 🎯 Filter by confidence ≥ 0.95
    mask = confidences >= 0.95
    if np.any(mask):
        print(f"\n📋 Classification Report (Confidence ≥ 0.95, {mask.sum()} samples):")
        print(classification_report(y_true[mask], y_pred[mask], digits=4))
    else:
        print("⚠️ No samples with confidence ≥ 0.95.")

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
        tips.append("☢️ Reduce alien toxicity: Isolate or neutralize compounds.")
    if sample["radiation_level"] > 0.8:
        tips.append("🛡 Shield Radiation: Add subsurface shielding or protective domes.")
    if sample["soc"] < 0.3:
        tips.append("🪵 Increase Organic Carbon: Mix compost or biomass.")
    if not tips:
        tips.append("✅ Soil is close to optimal. Minor improvements may suffice.")
    return tips

# 🔎 Show top 10 with verdict + tips
print("\n🔬 Top 10 Sample Predictions + Suggestions")
for i in range(min(10, len(y_pred))):
    pred_label = reverse_map[y_pred[i]]
    confidence = confidences[i]
    verdict = "✅ HABITABLE" if y_pred[i] == 2 else "❌ NOT HABITABLE"
    print(f"\nSample {i+1} → Pred={pred_label} | Confidence={confidence:.2f} → {verdict}")

    if y_pred[i] != 2:
        sample_features = dict(zip(required_columns, X[i]))
        for suggestion in suggest_improvements(sample_features):
            print("   💡", suggestion)

# 💾 Save output
output_df = df.copy()
output_df["Predicted_Label"] = [reverse_map[p] for p in y_pred]
output_df["Confidence"] = confidences
output_df.to_csv(args.output, index=False)
print(f"\n✅ Predictions saved to: {args.output}")
