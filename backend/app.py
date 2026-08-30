from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional
from scaling_bridge import get_model_input
from validator import validate_inputs
from explainability import get_feature_importance
from blockchain import (
    add_block, 
    verify_chain, 
    get_patient_history, 
    get_block_by_hash,
    get_chain_stats,
    determine_triage_level,
    get_chain
)
import joblib
import json
import numpy as np
import uuid
from datetime import datetime

app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://localhost:5174",
        "http://localhost:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5174"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load model + label encoder
model = joblib.load("model/medi_guard_merged_model.pkl")
label_encoder = joblib.load("model/label_encoder.pkl")

class RawInput(BaseModel):
    Glucose: float
    Cholesterol: float
    Hemoglobin: float
    Platelets: float
    White_Blood_Cells: float
    Red_Blood_Cells: float
    Hematocrit: float
    Mean_Corpuscular_Volume: float
    Mean_Corpuscular_Hemoglobin: float
    Mean_Corpuscular_Hemoglobin_Concentration: float
    Insulin: float
    BMI: float
    Systolic_Blood_Pressure: float
    Diastolic_Blood_Pressure: float
    Triglycerides: float
    HbA1c: float
    LDL_Cholesterol: float
    HDL_Cholesterol: float
    ALT: float
    AST: float
    Heart_Rate: float
    Creatinine: float
    Troponin: float
    C_reactive_Protein: float
    patient_id: Optional[str] = Field(None, description="Optional patient identifier. If not provided, a unique ID will be generated.")

@app.post("/predict")
def predict(data: RawInput):
    """
    Predict disease from raw clinical values.
    Accepts raw clinical values (e.g., Glucose: 120 mg/dL) and returns prediction.
    """
    print("\n" + "="*80)
    print(f"🩺 NEW PREDICTION REQUEST - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*80)
    
    raw_dict = data.dict()
    
    # Extract patient_id if present (it's optional and not a clinical value)
    patient_id = raw_dict.pop("patient_id", None)
    
    # Print received raw clinical values
    print("\n📥 RECEIVED RAW CLINICAL VALUES:")
    print("-" * 80)
    for key, value in raw_dict.items():
        if value is not None:
            try:
                print(f"  {key:40s}: {value:>10.2f}")
            except (TypeError, ValueError):
                print(f"  {key:40s}: {value}")
        else:
            print(f"  {key:40s}: None")
    
    errors = validate_inputs(raw_dict)

    if errors:
        print("\n❌ VALIDATION ERRORS:")
        for error in errors:
            print(f"  - {error}")
        print("="*80 + "\n")
        return {"status": "error", "errors": errors}

    try:
        print("\n🔄 NORMALIZING RAW VALUES TO MODEL INPUT FORMAT...")
        print("-" * 80)
        
        # Normalize raw clinical values to model input format (0-1 range)
        # get_model_input returns shape (1, 24) already
        scaled_2d = get_model_input(raw_dict, model_type="xgboost")
        
        print(f"✅ Normalization completed. Output shape: {scaled_2d.shape}")
        print(f"   Normalized value range: [{scaled_2d.min():.4f}, {scaled_2d.max():.4f}]")
        print(f"   First 5 normalized values: {scaled_2d[0][:5]}")
        print(f"   Total features: {scaled_2d.shape[1]} (expected: 24)")
        
        print("\n🤖 RUNNING ML MODEL PREDICTION...")
        print("-" * 80)
        
        # Make prediction
        prediction_encoded = model.predict(scaled_2d)[0]
        prediction_label = label_encoder.inverse_transform([prediction_encoded])[0]
        
        # Get prediction probabilities
        prediction_proba = model.predict_proba(scaled_2d)[0]
        probabilities = {
            label: float(prob) 
            for label, prob in zip(label_encoder.classes_, prediction_proba)
        }
        
        # Print prediction results
        print(f"🎯 PREDICTED DISEASE: {prediction_label}")
        print("\n📊 PREDICTION PROBABILITIES:")
        sorted_probs = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)
        for disease, prob in sorted_probs:
            bar_length = int(prob * 50)
            bar = "█" * bar_length
            print(f"  {disease:30s}: {prob*100:>6.2f}% {bar}")
        
        # Get feature importance
        importance = get_feature_importance(model)
        if importance:
            print("\n🔍 TOP 5 MOST IMPORTANT FEATURES:")
            # Expected feature order from scaling_bridge
            feature_names = [
                "Glucose", "Cholesterol", "Hemoglobin", "Platelets", "White_Blood_Cells",
                "Red_Blood_Cells", "Hematocrit", "Mean_Corpuscular_Volume",
                "Mean_Corpuscular_Hemoglobin", "Mean_Corpuscular_Hemoglobin_Concentration",
                "Insulin", "BMI", "Systolic_Blood_Pressure", "Diastolic_Blood_Pressure",
                "Triglycerides", "HbA1c", "LDL_Cholesterol", "HDL_Cholesterol",
                "ALT", "AST", "Heart_Rate", "Creatinine", "Troponin", "C_reactive_Protein"
            ]
            if len(importance) == len(feature_names):
                feature_importance_pairs = list(zip(feature_names, importance))
                feature_importance_pairs.sort(key=lambda x: x[1], reverse=True)
                for i, (feature, imp) in enumerate(feature_importance_pairs[:5], 1):
                    print(f"  {i}. {feature:40s}: {imp:.4f}")

        # Generate patient ID if not provided
        if not patient_id:
            patient_id = f"PAT_{uuid.uuid4().hex[:12].upper()}"
        
        # Determine triage level
        triage_level = determine_triage_level(prediction_label)
        
        # Blockchain logging - immutable record of AI prediction and triage
        blockchain_entry = add_block(patient_id, prediction_label, triage_level)
        
        print(f"\n🔗 BLOCKCHAIN ENTRY CREATED:")
        print(f"   Patient ID: {patient_id}")
        print(f"   Block Index: {blockchain_entry.get('index', 'N/A')}")
        print(f"   Hash: {blockchain_entry.get('hash', 'N/A')[:16]}...")
        print(f"   Previous Hash: {blockchain_entry.get('previous_hash', 'N/A')[:16]}...")
        print(f"   Triage Level: {triage_level}")
        print(f"   Timestamp: {datetime.fromtimestamp(blockchain_entry.get('timestamp', 0)).strftime('%Y-%m-%d %H:%M:%S')}")

        print("\n" + "="*80)
        print("✅ PREDICTION COMPLETE - Response sent to frontend")
        print("="*80 + "\n")

        return {
            "status": "success",
            "patient_id": patient_id,
            "prediction": prediction_label,
            "triage_level": triage_level,
            "probabilities": probabilities,
            "feature_importance": importance,
            "blockchain_entry": {
                "block_index": blockchain_entry.get("index"),
                "hash": blockchain_entry.get("hash"),
                "previous_hash": blockchain_entry.get("previous_hash"),
                "timestamp": blockchain_entry.get("timestamp"),
                "triage_level": blockchain_entry.get("triage_level")
            }
        }
    except Exception as e:
        print(f"\n❌ PREDICTION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        print("="*80 + "\n")
        return {
            "status": "error",
            "message": f"Prediction failed: {str(e)}"
        }

@app.get("/blockchain/verify")
def verify_blockchain():
    """
    Verify the integrity of the entire blockchain.
    Returns validation status and any errors found.
    """
    is_valid, errors = verify_chain()
    return {
        "status": "valid" if is_valid else "invalid",
        "is_valid": is_valid,
        "errors": errors,
        "message": "Blockchain is valid and immutable" if is_valid else "Blockchain integrity check failed"
    }

@app.get("/blockchain/stats")
def get_blockchain_stats():
    """
    Get statistics about the blockchain.
    Returns total blocks, patients, triage distribution, etc.
    """
    stats = get_chain_stats()
    return {
        "status": "success",
        "stats": stats
    }

@app.get("/blockchain/patient/{patient_id}")
def get_patient_blockchain_history(patient_id: str):
    """
    Retrieve all blockchain entries for a specific patient.
    Creates an auditable trail of all AI predictions for that patient.
    """
    history = get_patient_history(patient_id)
    if not history:
        raise HTTPException(status_code=404, detail=f"No blockchain entries found for patient {patient_id}")
    
    # Format timestamps for readability
    formatted_history = []
    for block in history:
        formatted_block = block.copy()
        formatted_block["timestamp_readable"] = datetime.fromtimestamp(block["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
        formatted_history.append(formatted_block)
    
    return {
        "status": "success",
        "patient_id": patient_id,
        "total_entries": len(history),
        "history": formatted_history
    }

@app.get("/blockchain/block/{block_hash}")
def get_block_by_hash_endpoint(block_hash: str):
    """
    Retrieve a specific block by its hash.
    Useful for verifying individual records.
    """
    block = get_block_by_hash(block_hash)
    if not block:
        raise HTTPException(status_code=404, detail=f"Block with hash {block_hash[:16]}... not found")
    
    formatted_block = block.copy()
    formatted_block["timestamp_readable"] = datetime.fromtimestamp(block["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
    
    return {
        "status": "success",
        "block": formatted_block
    }

@app.get("/blockchain/audit")
def audit_blockchain():
    """
    Comprehensive audit of the blockchain.
    Returns full chain with verification status.
    """
    chain = get_chain()
    is_valid, errors = verify_chain()
    stats = get_chain_stats()
    
    # Format chain for readability
    formatted_chain = []
    for block in chain:
        formatted_block = block.copy()
        formatted_block["timestamp_readable"] = datetime.fromtimestamp(block["timestamp"]).strftime('%Y-%m-%d %H:%M:%S')
        formatted_chain.append(formatted_block)
    
    return {
        "status": "success",
        "chain_valid": is_valid,
        "verification_errors": errors,
        "statistics": stats,
        "chain": formatted_chain
    }
