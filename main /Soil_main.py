import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report
import tensorflow as tf
import gym
from gym import spaces

# ===============================
# 1. Custom Gym Environment
# ===============================
class AlienSoilEnv(gym.Env):
    def __init__(self, csv_path):
        super(AlienSoilEnv, self).__init__()
        self.df = pd.read_csv(csv_path)

        # Map labels
        self.label_map = {"not_habitable": 0, "moderate": 1, "habitat": 2}
        self.reverse_map = {v: k for k, v in self.label_map.items()}
        self.df['label'] = self.df['habitat'].map(self.label_map)

        # Features used in RL state
        self.features = [
            'phh2o', 'soc', 'clay', 'sand', 'silt', 'cec',
            'bdod', 'nitrogen', 'radiation_level', 'alien_chemical_index'
        ]

        # Observation space = soil feature ranges
        self.observation_space = spaces.Box(
            low=np.array([self.df[feature].min() for feature in self.features]),
            high=np.array([self.df[feature].max() for feature in self.features]),
            dtype=np.float32
        )

        # Actions = predict habitability (0,1,2)
        self.action_space = spaces.Discrete(3)

        self.current_index = 0
        self.max_index = len(self.df)

    def reset(self):
        self.current_index = np.random.randint(0, self.max_index)
        sample = self.df.iloc[self.current_index]
        self.state = sample[self.features].values.astype(np.float32)
        self.correct_label = sample['label']
        return self.state

    def step(self, action):
        done = True
        reward = 1.0 if action == self.correct_label else -0.1
        info = {
            "correct_label": self.correct_label,
            "action_taken": action,
        }
        return self.state, reward, done, info

    def render(self, mode='human'):
        print("Current State:", self.state)
        print("Correct Label:", self.correct_label)


# ===============================
# 2. Evaluation + Suggestion Engine
# ===============================
def evaluate_and_suggest(csv_path, model_path):
    print("📦 Loading dataset...")
    mars_df = pd.read_csv(csv_path)

    label_map = {"not_habitable": 0, "moderate": 1, "habitat": 2}
    reverse_map = {v: k for k, v in label_map.items()}
    mars_df["habitat"] = mars_df["habitat"].map(label_map)

    # Fix depth column if string
    if "depth" in mars_df.columns and mars_df["depth"].dtype == object:
        mars_df["depth"] = mars_df["depth"].str.extract(r"(\d+)").astype(float)

    required_columns = [
        "lat", "lon", "depth", "phh2o", "soc", "clay", "sand", "silt", "cec",
        "bdod", "nitrogen", "radiation_level", "alien_chemical_index",
        "reward_score", "confidence"
    ]
    for col in required_columns:
        if col not in mars_df.columns:
            mars_df[col] = 0.0

    X = mars_df[required_columns].values
    y_true = mars_df["habitat"].values

    # Normalize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Load model
    print(f"📥 Loading model from {model_path}...")
    model = tf.keras.models.load_model(model_path)

    # Predictions
    print("🧠 Making predictions...")
    pred_probs = model.predict(X_scaled)
    y_pred = np.argmax(pred_probs, axis=1)
    confidences = np.max(pred_probs, axis=1)

    # Reports
    print("\n📋 Classification Report (Full Dataset):")
    print(classification_report(y_true, y_pred, digits=4))

    mask = confidences >= 0.95
    if np.any(mask):
        print(f"\n📋 Classification Report (Confidence ≥ 0.95, {mask.sum()} samples):")
        print(classification_report(y_true[mask], y_pred[mask], digits=4))
    else:
        print("⚠️ No samples with confidence ≥ 0.95.")

    # Suggestion engine
    def suggest_improvements(sample):
        tips = []
        if sample["phh2o"] < 6.0:
            tips.append("🧪 Raise pH: Add alkaline minerals (lime).")
        elif sample["phh2o"] > 8.0:
            tips.append("🧪 Lower pH: Add sulfur compounds or organic matter.")
        if sample["nitrogen"] < 0.5:
            tips.append("🌱 Boost Nitrogen: Add nitrogen-fixing microbes or ammonium salts.")
        if sample["alien_chemical_index"] > 0.6:
            tips.append("☢️ Reduce alien toxicity: Isolate or neutralize compounds.")
        if sample["radiation_level"] > 0.8:
            tips.append("🛡 Shield Radiation: Subsurface shielding or domes.")
        if sample["soc"] < 0.3:
            tips.append("🪵 Increase Organic Carbon: Mix composted biomass or carbon-rich regolith.")
        if not tips:
            tips.append("✅ Soil is close to optimal.")
        return tips

    print("\n🔬 Top 10 Sample Predictions + Suggestions")
    for i in range(min(10, len(y_pred))):
        pred_label = reverse_map[y_pred[i]]
        true_label = reverse_map[y_true[i]]
        confidence = confidences[i]
        verdict = "✅ HABITABLE" if y_pred[i] == 2 else "❌ NOT HABITABLE"
        print(f"\nSample {i+1} → True={true_label} | Pred={pred_label} | Confidence={confidence:.2f} → {verdict}")

        if y_pred[i] != 2:
            sample_features = dict(zip(required_columns, X[i]))
            for suggestion in suggest_improvements(sample_features):
                print("   💡", suggestion)

    # Save results
    output_df = mars_df.copy()
    output_df["Predicted_Label"] = [reverse_map[p] for p in y_pred]
    output_df["Confidence"] = confidences
    output_csv = "alien_soil_predictions.csv"
    output_df.to_csv(output_csv, index=False)
    print(f"\n✅ All predictions saved to: {output_csv}")


# ===============================
# 3. Main runner
# ===============================
if __name__ == "__main__":
    csv_path = r"C:\Work_place\isru project v1\data\simulated_mars_soil_dataset.csv"
    model_path = r"C:\Work_place\isru project v1\models\alien_soil_model_ep10000.keras"

    # Init RL Environment
    env = AlienSoilEnv(csv_path)
    print("✅ Gym environment ready! (state space:", env.observation_space, ")")

    # Evaluate + Suggest improvements
    evaluate_and_suggest(csv_path, model_path)
