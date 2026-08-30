import pandas as pd
import numpy as np
import joblib
import os

# ----------------------------------------------------
# 1. SETUP & LOADING
# ----------------------------------------------------
# Corrected model name from your previous step
MODEL_PATH = "model/medi_guard_merged_model.pkl"
ENCODER_PATH = "model/label_encoder.pkl"
DATA_PATH = "data/test.csv"  # Use test data for "unseen" base

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at {MODEL_PATH}")

print("Loading model and data...")
model = joblib.load(MODEL_PATH)
label_encoder = joblib.load(ENCODER_PATH)
df = pd.read_csv(DATA_PATH)

# ----------------------------------------------------
# 2. CREATE A "NEW" REALISTIC PATIENT
# ----------------------------------------------------
# Pick a random real patient to start with
random_idx = np.random.randint(0, len(df))
real_patient = df.iloc[random_idx]
true_disease = real_patient["Disease"]

# Separate features
features = real_patient.drop("Disease")

print(f"\nOriginal Patient Profile (Row {random_idx}):")
print(f"True Condition: {true_disease}")
print("-" * 40)

# Create a "Variation" of this patient
# We add random noise (+/- 10%) to make the values unique/new
# This keeps the biological correlations intact (High Glucose stays relatively High)
print("Generatng unique variation (Perturbation +/- 10%)...")

synthetic_patient = {}
for col, val in features.items():
    # Calculate noise: random value between -10% and +10% of the original value
    noise = val * np.random.uniform(-0.10, 0.10)
    new_val = val + noise
    synthetic_patient[col] = round(new_val, 6)

# Convert to DataFrame for prediction
sample_df = pd.DataFrame([synthetic_patient])

# ----------------------------------------------------
# 3. PREDICT & VERIFY
# ----------------------------------------------------
# Ensure column order matches training
if hasattr(model, "feature_names_in_"):
    sample_df = sample_df[model.feature_names_in_]

encoded_pred = model.predict(sample_df)[0]
predicted_label = label_encoder.inverse_transform([encoded_pred])[0]

print("\n================ TEST RESULTS ================")
print("Generated 'New' Patient Values (Snippet):")
print(sample_df.iloc[0].head(5).to_string()) 
print("...")
print("-" * 40)
print(f"Original Disease:   {true_disease}")
print(f"Predicted Disease:  {predicted_label}")
print("==============================================")

if true_disease == predicted_label:
    print("✅ SUCCESS: Model correctly identified the modified patient.")
else:
    print("❌ WARNING: Prediction changed! The variation confused the model.")