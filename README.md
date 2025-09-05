🌌 Alien Soil Doctor (ISRU Project)

Alien Soil Doctor is an AI-powered project designed to evaluate alien soil habitability by extending Earth-based soil datasets with extraterrestrial parameters such as radiation levels, alien chemical index, and toxic compound risks.
It is part of an In-Situ Resource Utilization (ISRU) research initiative and uses Reinforcement Learning (RL) for continual self-improvement.

🚀 Features

✅ Dataset preprocessing & balancing

✅ Alien soil simulation environment

✅ Training pipeline with TensorFlow/Keras

✅ Reinforcement Learning (RL) for continual learning

✅ Interactive UI for results & suggestions

✅ Execution batch file for one-click running

📂 Project Structure
Alien-Soil-Doctor/
│
├── main/
│   ├── Balanced_csv.py        # Balances the dataset for training/evaluation
│   ├── Soil_main.py           # Main script: loads model, predicts, and evaluates
│   ├── UI.py                  # Interactive UI (CLI-based) for results & tips
│   ├── env.py                 # Custom Gym environment for RL training
│   └── model_training.py      # Model training pipeline (TensorFlow/Keras)
│
├── model/                     # Stores trained models (.keras, checkpoints)
│
├── alien_soil_doctor_run.bat  # Batch script to execute the project
├── requirements.txt           # Python dependencies
└── README.md                  # Documentation

📑 Dataset Requirements

Your input dataset should be a CSV file with the following required columns:

| Column                 | Type         | Description                                           |
| ---------------------- | ------------ | ----------------------------------------------------- |
| `lat`                  | float        | Latitude coordinate                                   |
| `lon`                  | float        | Longitude coordinate                                  |
| `depth`                | float/string | Soil depth (numeric or ranges like `"0-30cm"`)        |
| `phh2o`                | float        | Soil pH in water                                      |
| `soc`                  | float        | Soil organic carbon                                   |
| `clay`                 | float        | Clay %                                                |
| `sand`                 | float        | Sand %                                                |
| `silt`                 | float        | Silt %                                                |
| `cec`                  | float        | Cation exchange capacity                              |
| `bdod`                 | float        | Bulk density                                          |
| `nitrogen`             | float        | Nitrogen concentration                                |
| `radiation_level`      | float        | Extraterrestrial radiation index (0–1)                |
| `alien_chemical_index` | float        | Alien compound index (0–1)                            |
| `reward_score`         | float        | RL feedback reward score                              |
| `confidence`           | float        | Confidence of sample                                  |
| `habitat`              | string       | Target label → `habitat`, `moderate`, `not_habitable` |

⚡ Note: If a required column is missing, scripts like Soil_main.py will auto-fill it with defaults (0.0).

⚙️ Installation

Clone this repository:

git clone https://github.com/<your-username>/Alien-Soil-Doctor.git
cd Alien-Soil-Doctor


Install dependencies:

pip install -r requirements.txt


Place your dataset in data/ (e.g., simulated_mars_soil_dataset.csv).

▶️ Execution

Run the full pipeline with:

alien_soil_doctor_run.bat


This will:

Preprocess & balance dataset (Balanced_csv.py)

Train or load model (model_training.py)

Evaluate soil habitability (Soil_main.py)

Run UI for predictions & improvement tips (UI.py)

Save results to alien_soil_predictions.csv

🖥️ Example Output
📦 Loading dataset...
📥 Model loaded from /model/alien_soil_model_ep10000.keras
🧠 Making predictions...

Sample 1 → True=moderate | Pred=not_habitable | Confidence=0.89 → ❌ NOT HABITABLE
   💡 Raise pH: Add alkaline minerals (lime).
   💡 Boost Nitrogen: Add nitrogen-fixing microbes.

📜 Batch File (alien_soil_doctor_run.bat)
@echo off
echo 🌌 Running Alien Soil Doctor...
python main/Soil_main.py
pause


Double-click this file to run the project end-to-end.

🔮 Future Scope

Integration with real datasets (NASA/ESA/ISRO)

Deployable edge AI for onboard spacecraft soil testing

More advanced RL agents (PPO/DQN with GPU acceleration)

Live soil analysis through robotic probes

👨‍🚀 Author

Mridul Gupta
🚀 AI & Robotics Enthusiast | ISRU Researcher
