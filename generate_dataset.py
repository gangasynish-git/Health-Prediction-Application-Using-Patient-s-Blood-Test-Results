import pandas as pd
import numpy as np

np.random.seed(42)

rows = 5000

glucose = np.random.randint(70, 250, rows)
haemoglobin = np.round(np.random.uniform(8, 18, rows), 1)
cholesterol = np.random.randint(120, 320, rows)

risk = []

for g, h, c in zip(glucose, haemoglobin, cholesterol):

    score = 0

    if g > 140:
        score += 1

    if h < 12:
        score += 1

    if c > 200:
        score += 1

    if score == 0:
        risk.append("Healthy")

    elif score == 1:
        risk.append("Moderate Risk")

    else:
        risk.append("High Risk")

df = pd.DataFrame({
    "Glucose": glucose,
    "Haemoglobin": haemoglobin,
    "Cholesterol": cholesterol,
    "Risk": risk
})

df.to_csv("health_dataset.csv", index=False)

print("Dataset created successfully")
print(df.head())