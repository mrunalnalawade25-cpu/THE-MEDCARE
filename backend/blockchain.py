import hashlib
import time
import json
import os
from datetime import datetime
from typing import Dict, List, Optional, Tuple

CHAIN_FILE = "chain.json"

# Triage severity levels based on disease predictions
TRIAGE_LEVELS = {
    "Critical": ["Heart Disease", "Heart Di", "Diabetes", "Hypertension"],
    "High": ["Anemia", "Thalassemia", "Thalasse"],
    "Medium": ["Chronic Kidney Disease", "Liver Disease"],
    "Low": ["Healthy", "Normal"]
}

def determine_triage_level(prediction: str) -> str:
    """
    Determine triage priority level based on disease prediction.
    Returns: 'Critical', 'High', 'Medium', or 'Low'
    """
    prediction_clean = prediction.strip().replace("'", "")
    
    for level, diseases in TRIAGE_LEVELS.items():
        if any(disease.lower() in prediction_clean.lower() for disease in diseases):
            return level
    
    # Default to Medium if unknown
    return "Medium"

def calculate_hash(block_data: Dict) -> str:
    """
    Calculate SHA-256 hash of a block's data.
    Includes all block fields to ensure immutability.
    """
    block_string = json.dumps(block_data, sort_keys=True)
    return hashlib.sha256(block_string.encode()).hexdigest()

def migrate_old_chain(old_chain: List[Dict]) -> List[Dict]:
    """
    Migrate old chain format to new format with proper linking.
    Old format: {patient_id, prediction, timestamp, hash}
    New format: {index, patient_id, prediction, triage_level, timestamp, previous_hash, hash, nonce}
    """
    if not old_chain:
        return []
    
    migrated_chain = []
    
    # Check if first block is already in new format
    if "index" in old_chain[0] and "previous_hash" in old_chain[0]:
        return old_chain  # Already migrated
    
    # Create genesis block
    genesis = get_genesis_block()
    migrated_chain.append(genesis)
    
    # Migrate old blocks
    previous_hash = genesis["hash"]
    for i, old_block in enumerate(old_chain, start=1):
        # Determine triage level
        triage_level = determine_triage_level(old_block.get("prediction", ""))
        
        # Create new block structure
        new_block = {
            "index": i,
            "patient_id": old_block.get("patient_id", "UNKNOWN"),
            "prediction": old_block.get("prediction", ""),
            "triage_level": triage_level,
            "timestamp": old_block.get("timestamp", time.time()),
            "previous_hash": previous_hash,
            "nonce": 0
        }
        
        # Recalculate hash with new structure
        new_block["hash"] = calculate_hash(new_block)
        migrated_chain.append(new_block)
        previous_hash = new_block["hash"]
    
    return migrated_chain

def get_chain() -> List[Dict]:
    """Load the blockchain from file and migrate if necessary."""
    if os.path.exists(CHAIN_FILE):
        try:
            with open(CHAIN_FILE, "r") as f:
                chain = json.load(f)
                # Ensure chain is a list
                if not isinstance(chain, list):
                    return []
                
                # Migrate if old format detected
                if chain and ("index" not in chain[0] or "previous_hash" not in chain[0]):
                    print("⚠️  Migrating old blockchain format to new format...")
                    chain = migrate_old_chain(chain)
                    save_chain(chain)
                    print("✅ Blockchain migration completed")
                
                return chain
        except (json.JSONDecodeError, IOError):
            return []
    return []

def save_chain(chain: List[Dict]) -> None:
    """Save the blockchain to file."""
    with open(CHAIN_FILE, "w") as f:
        json.dump(chain, f, indent=4)

def get_genesis_block() -> Dict:
    """Create the genesis (first) block of the chain."""
    timestamp = time.time()
    block_data = {
        "index": 0,
        "patient_id": "GENESIS",
        "prediction": "GENESIS_BLOCK",
        "triage_level": "N/A",
        "timestamp": timestamp,
        "previous_hash": "0" * 64,  # 64 zeros for genesis block
        "nonce": 0
    }
    block_data["hash"] = calculate_hash(block_data)
    return block_data

def add_block(patient_id: str, prediction: str, triage_level: Optional[str] = None) -> Dict:
    """
    Add a new block to the blockchain with proper linking.
    
    Args:
        patient_id: Unique identifier for the patient
        prediction: AI's disease prediction
        triage_level: Optional triage level (auto-determined if not provided)
    
    Returns:
        The newly created block
    """
    # Load existing chain
    chain = get_chain()
    
    # Initialize chain with genesis block if empty
    if not chain:
        chain = [get_genesis_block()]
    
    # Determine triage level if not provided
    if triage_level is None:
        triage_level = determine_triage_level(prediction)
    
    # Get previous block
    previous_block = chain[-1]
    previous_hash = previous_block["hash"]
    
    # Create new block
    timestamp = time.time()
    block_index = len(chain)
    
    block_data = {
        "index": block_index,
        "patient_id": patient_id,
        "prediction": prediction,
        "triage_level": triage_level,
        "timestamp": timestamp,
        "previous_hash": previous_hash,
        "nonce": 0
    }
    
    # Calculate hash (includes previous_hash for immutability)
    block_data["hash"] = calculate_hash(block_data)
    
    # Add block to chain
    chain.append(block_data)
    save_chain(chain)
    
    return block_data

def verify_chain() -> Tuple[bool, List[str]]:
    """
    Verify the integrity of the entire blockchain.
    
    Returns:
        Tuple of (is_valid, list_of_errors)
    """
    chain = get_chain()
    errors = []
    
    if not chain:
        return True, []  # Empty chain is valid
    
    # Check genesis block
    if chain[0]["index"] != 0 or chain[0]["previous_hash"] != "0" * 64:
        errors.append("Genesis block is invalid")
        return False, errors
    
    # Verify each block
    for i in range(1, len(chain)):
        current_block = chain[i]
        previous_block = chain[i - 1]
        
        # Verify index sequence
        if current_block["index"] != i:
            errors.append(f"Block {i} has incorrect index")
        
        # Verify previous hash link
        if current_block["previous_hash"] != previous_block["hash"]:
            errors.append(f"Block {i} has broken chain link (previous_hash mismatch)")
        
        # Verify block hash
        # Create a copy without hash to recalculate
        block_copy = current_block.copy()
        stored_hash = block_copy.pop("hash")
        calculated_hash = calculate_hash(block_copy)
        
        if stored_hash != calculated_hash:
            errors.append(f"Block {i} hash is invalid (data tampered)")
    
    return len(errors) == 0, errors

def get_patient_history(patient_id: str) -> List[Dict]:
    """
    Retrieve all blockchain entries for a specific patient.
    
    Args:
        patient_id: Patient identifier
    
    Returns:
        List of blocks for the patient
    """
    chain = get_chain()
    return [block for block in chain if block.get("patient_id") == patient_id]

def get_block_by_hash(block_hash: str) -> Optional[Dict]:
    """
    Retrieve a block by its hash.
    
    Args:
        block_hash: Hash of the block to retrieve
    
    Returns:
        Block dictionary or None if not found
    """
    chain = get_chain()
    for block in chain:
        if block.get("hash") == block_hash:
            return block
    return None

def get_chain_stats() -> Dict:
    """
    Get statistics about the blockchain.
    
    Returns:
        Dictionary with chain statistics
    """
    chain = get_chain()
    if not chain:
        return {
            "total_blocks": 0,
            "total_patients": 0,
            "chain_valid": True,
            "triage_distribution": {}
        }
    
    is_valid, _ = verify_chain()
    patient_ids = set(block.get("patient_id") for block in chain if block.get("patient_id") != "GENESIS")
    
    triage_distribution = {}
    for block in chain:
        triage = block.get("triage_level", "Unknown")
        triage_distribution[triage] = triage_distribution.get(triage, 0) + 1
    
    return {
        "total_blocks": len(chain),
        "total_patients": len(patient_ids),
        "chain_valid": is_valid,
        "triage_distribution": triage_distribution,
        "last_block_timestamp": chain[-1].get("timestamp") if chain else None
    }
