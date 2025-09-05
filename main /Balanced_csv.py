import pandas as pd

# Path to original Earth soil dataset
path = "C:/Work_place/isru project final/updated_soil_dataset.csv"

# Load it
df = pd.read_csv(path)

# Balance classes: 200k moderate, 200k not_habitable, all habitat (~10k)
df_balanced = pd.concat([
    df[df["habitat"] == "moderate"].sample(200000, random_state=42),
    df[df["habitat"] == "not_habitable"].sample(200000, random_state=42),
    df[df["habitat"] == "habitat"]
])

# Shuffle
df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
output_path = "C:/Work_place/isru project final/balanced_soil_dataset.csv"
df_balanced.to_csv(output_path, index=False)

# Check result
print("✅ Balanced dataset saved!")
print(df_balanced["habitat"].value_counts())
