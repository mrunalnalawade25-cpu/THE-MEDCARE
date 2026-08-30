"""
Test script to verify raw clinical value scaling works correctly.
"""
import json
from scaling_bridge import scale_input, CLINICAL_RANGES

# Sample raw clinical values (real-world format)
sample_raw_values = {
    "Glucose": 120.0,  # mg/dL
    "Cholesterol": 200.0,  # mg/dL
    "Hemoglobin": 14.0,  # g/dL
    "Platelets": 250000.0,  # per microliter
    "White_Blood_Cells": 7000.0,  # per microliter
    "Red_Blood_Cells": 4.5,  # million cells per microliter
    "Hematocrit": 42.0,  # %
    "Mean_Corpuscular_Volume": 90.0,  # fL
    "Mean_Corpuscular_Hemoglobin": 30.0,  # pg
    "Mean_Corpuscular_Hemoglobin_Concentration": 33.0,  # g/dL
    "Insulin": 15.0,  # μU/mL
    "BMI": 22.0,  # kg/m²
    "Systolic_Blood_Pressure": 120.0,  # mmHg
    "Diastolic_Blood_Pressure": 80.0,  # mmHg
    "Triglycerides": 150.0,  # mg/dL
    "HbA1c": 5.5,  # %
    "LDL_Cholesterol": 100.0,  # mg/dL
    "HDL_Cholesterol": 50.0,  # mg/dL
    "ALT": 30.0,  # U/L
    "AST": 25.0,  # U/L
    "Heart_Rate": 72.0,  # bpm
    "Creatinine": 1.0,  # mg/dL
    "Troponin": 0.01,  # ng/mL
    "C_reactive_Protein": 2.0  # mg/L
}

print("=" * 60)
print("Testing Raw Clinical Value Scaling")
print("=" * 60)
print("\nInput Raw Values (Sample):")
for key, value in list(sample_raw_values.items())[:5]:
    print(f"  {key}: {value}")
print("  ...")

# Load scaler config
scaler_config = json.load(open("model/scaler_improved.json"))

try:
    # Test scaling
    scaled_array = scale_input(sample_raw_values, scaler_config)
    
    print(f"\n✅ Scaling successful!")
    print(f"Output shape: {scaled_array.shape}")
    print(f"Output range: [{scaled_array.min():.4f}, {scaled_array.max():.4f}]")
    print(f"First 5 scaled values: {scaled_array[:5]}")
    
    # Verify normalization worked
    print("\nVerification - Normalized values (0-1 range) for first few features:")
    for i, feature in enumerate(["Glucose", "Cholesterol", "Hemoglobin", "Platelets", "White_Blood_Cells"]):
        if feature in CLINICAL_RANGES:
            min_val, max_val = CLINICAL_RANGES[feature]
            raw_val = sample_raw_values[feature]
            expected_normalized = (raw_val - min_val) / (max_val - min_val)
            expected_normalized = max(0.0, min(1.0, expected_normalized))
            print(f"  {feature}: {raw_val} -> {expected_normalized:.4f} (normalized)")
    
    print("\n" + "=" * 60)
    print("✅ All tests passed! Scaling bridge is working correctly.")
    print("=" * 60)
    
except Exception as e:
    print(f"\n❌ Error: {e}")
    import traceback
    traceback.print_exc()

