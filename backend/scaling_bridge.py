import joblib
import numpy as np
import os

# ---------------------------------------------------------
# 1. SETUP PATHS & LOAD SCALER
# ---------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
SCALER_PATH = os.path.join(BASE_DIR, "model", "standard_scaler.pkl")

saved_scaler = None

# Try to load the scaler
try:
    if os.path.exists(SCALER_PATH):
        saved_scaler = joblib.load(SCALER_PATH)
    else:
        # Check parent directory fallback
        PARENT_PATH = os.path.join(BASE_DIR, "..", "model", "standard_scaler.pkl")
        if os.path.exists(PARENT_PATH):
            saved_scaler = joblib.load(PARENT_PATH)
            SCALER_PATH = PARENT_PATH
        else:
            print("⚠️ WARNING: Scaler not found. KNN predictions will fail.")
except Exception as e:
    print(f"Error loading scaler: {e}")

# ---------------------------------------------------------
# 2. DEFINE CLINICAL RANGES (REQUIRED FOR NORMALIZATION)
# ---------------------------------------------------------
# These match the Min/Max used to create your 0-1 training data.
CLINICAL_RANGES = {
    "Glucose": (50, 250),           # LOWERED Max from 500 -> 250 (Makes 185 a "High" 0.67 instead of "Low" 0.3)
    "Cholesterol": (100, 350),      # Tightened
    "Hemoglobin": (5, 18),
    "Platelets": (50000, 450000),   # Standardized
    "White_Blood_Cells": (2000, 20000),
    "Red_Blood_Cells": (2, 7),
    "Hematocrit": (20, 55),
    "Mean_Corpuscular_Volume": (60, 110),
    "Mean_Corpuscular_Hemoglobin": (15, 35),
    "Mean_Corpuscular_Hemoglobin_Concentration": (25, 38),
    "Insulin": (2, 40),             # Tightened to catch resistance earlier
    "BMI": (15, 45),
    "Systolic_Blood_Pressure": (80, 180),
    "Diastolic_Blood_Pressure": (40, 110),
    "Triglycerides": (50, 500),     # LOWERED Max from 1000 -> 500 (Makes 220 more significant)
    "HbA1c": (3, 10),               # LOWERED Max from 15 -> 10 (Makes 8.2 a very high 0.74)
    "LDL_Cholesterol": (50, 250),
    "HDL_Cholesterol": (20, 100),
    "ALT": (5, 150),
    "AST": (5, 150),
    "Heart_Rate": (40, 120),
    "Creatinine": (0.5, 5),
    "Troponin": (0, 10),            # Troponin is usually very low, max 10 covers emergencies
    "C_reactive_Protein": (0, 20)
}

def normalize_value(key, value):
    """Converts raw value (e.g. 105) to 0-1 range based on clinical limits."""
    if key not in CLINICAL_RANGES:
        return value # Fallback
        
    min_val, max_val = CLINICAL_RANGES[key]
    
    # Formula: (x - min) / (max - min)
    normalized = (value - min_val) / (max_val - min_val)
    
    # Clip to ensure it stays between 0 and 1
    return max(0.0, min(1.0, normalized))

# ---------------------------------------------------------
# 3. MAIN PROCESSING FUNCTION
# ---------------------------------------------------------

def get_model_input(raw_data, model_type="xgboost"):
    """
    Args:
        raw_data: dict of RAW clinical values {"Glucose": 105, ...}
        model_type: "xgboost" or "knn"
    """
    
    EXPECTED_FEATURES_ORDER = [
        "Glucose", "Cholesterol", "Hemoglobin", "Platelets", "White_Blood_Cells",
        "Red_Blood_Cells", "Hematocrit", "Mean_Corpuscular_Volume",
        "Mean_Corpuscular_Hemoglobin", "Mean_Corpuscular_Hemoglobin_Concentration",
        "Insulin", "BMI", "Systolic_Blood_Pressure", "Diastolic_Blood_Pressure",
        "Triglycerides", "HbA1c", "LDL_Cholesterol", "HDL_Cholesterol",
        "ALT", "AST", "Heart_Rate", "Creatinine", "Troponin", "C_reactive_Protein"
    ]
    
    # 1. Normalize Raw Values -> 0-1 Range
    normalized_features = []
    for feature in EXPECTED_FEATURES_ORDER:
        raw_val = float(raw_data.get(feature, 0))
        norm_val = normalize_value(feature, raw_val)
        normalized_features.append(norm_val)
    
    # Reshape for model (1 sample, 24 features)
    # This 'X_normalized' now matches the format of your train.csv
    X_normalized = np.array(normalized_features).reshape(1, -1)

    # 2. Return based on Model Requirement
    if model_type == "xgboost":
        # XGBoost was trained on the 0-1 CSV data directly.
        return X_normalized
        
    elif model_type == "knn":
        # KNN was trained on StandardScaled version of the 0-1 data.
        if saved_scaler is None:
            raise ValueError("Scaler not loaded.")
            
        # Transform the 0-1 data into Standard Scale
        X_scaled = saved_scaler.transform(X_normalized)
        return X_scaled
        
    else:
        raise ValueError(f"Unknown model type: {model_type}")

# --- Test Block ---
if __name__ == "__main__":
    print("--- Testing Normalization Logic ---")
    test_data = {"Glucose": 105, "Cholesterol": 200}
    
    # Manually check calculation for Glucose (Range 50-500)
    # (105 - 50) / (500 - 50) = 55 / 450 = 0.122
    
    output = get_model_input(test_data, "xgboost")
    print(f"Raw Glucose: {test_data['Glucose']}")
    print(f"Normalized Input to Model: {output[0][0]:.4f}") 
    
    if 0.12 < output[0][0] < 0.13:
        print("✅ Correct! 105 converted to approx 0.12")
    else:
        print("❌ Incorrect normalization.")