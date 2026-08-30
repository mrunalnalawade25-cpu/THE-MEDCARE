import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from xgboost import XGBClassifier
import joblib
import warnings

warnings.filterwarnings("ignore")

print("===================================================")
print("      MEDI-GUARD MODEL TRAINING (train_test.csv)      ")
print("===================================================\n")

# ----------------------------------------------------
# 1. LOAD DATASET (ONLY train_test.csv)
# ----------------------------------------------------
print("[1] Loading train_test.csv ...")

# Path handling
if os.path.exists("data/new_train.csv"):
    df = pd.read_csv("data/new_train.csv")
else:
    df = pd.read_csv("data/new_train.csv")

print(f"Dataset Shape: {df.shape}")
print("Class Distribution:")
print(df["Disease"].value_counts(), "\n")

# ----------------------------------------------------
# 2. PREPARE FEATURES & TARGET
# ----------------------------------------------------
X = df.drop("Disease", axis=1)
y = df["Disease"]

label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y)

# ----------------------------------------------------
# 3. TRAIN-TEST SPLIT
# ----------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

print(f"Training size: {len(X_train)}")
print(f"Testing size: {len(X_test)}\n")

# ----------------------------------------------------
# 4. TRAIN XGBOOST MODEL (WITH EARLY STOPPING)
# ----------------------------------------------------
print("[2] Training XGBoost model with Early Stopping...")

model = XGBClassifier(
    n_estimators=2000,
    max_depth=7,
    learning_rate=0.05,
    subsample=0.9,
    colsample_bytree=0.9,
    eval_metric="mlogloss",
    random_state=42,
    early_stopping_rounds=50
)

model.fit(
    X_train, y_train,
    eval_set=[(X_test, y_test)],
    verbose=100
)

# ----------------------------------------------------
# 5. EVALUATION
# ----------------------------------------------------
print("\n================ MODEL PERFORMANCE ================\n")

pred = model.predict(X_test)
acc = accuracy_score(y_test, pred)

print(f"🎯 Test Accuracy: {acc*100:.2f}%")
print(f"Optimal Trees Used: {model.best_ntree_limit if hasattr(model, 'best_ntree_limit') else 'N/A'}\n")

print("Classification Report:")
print(classification_report(y_test, pred, target_names=label_encoder.classes_))

# ----------------------------------------------------
# 6. SAVE MODEL
# ----------------------------------------------------
print("\n[3] Saving model and label encoder...\n")

os.makedirs("model", exist_ok=True)
joblib.dump(model, "model/medi_guard_model.pkl")
joblib.dump(label_encoder, "model/label_encoder.pkl")

print("===================================================")
print("           MODEL SAVED SUCCESSFULLY!")
print("===================================================\n")
