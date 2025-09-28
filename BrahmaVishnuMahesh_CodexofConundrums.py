# ============================
# IMPORT REQUIRED LIBRARIES
# ============================
import os               # For interacting with file system
import re               # For regex operations (used in text cleaning & sorting)
import json             # For reading JSON files
import pandas as pd     # For handling data in tabular form (DataFrames)
from sklearn.model_selection import train_test_split  # Split dataset into train/validation
from sklearn.feature_extraction.text import TfidfVectorizer  # Convert text to numeric vectors
from sklearn.linear_model import LogisticRegression       # Logistic Regression classifier
from sklearn.metrics import classification_report          # For evaluating model performance
import numpy as np       # For numerical operations

# ============================
# 1. LOAD TRAINING DATA
# ============================
train_dir = r"C:\Users\adity\OneDrive\Desktop\ML chalange\archive\Final\train"  # Training dataset folder
data = []  # List to hold all structured data

# Loop through each category folder (algebra, geometry, etc.)
for category in os.listdir(train_dir):
    category_path = os.path.join(train_dir, category)
    if os.path.isdir(category_path):  # Ensure it's a folder
        for file_name in os.listdir(category_path):
            if file_name.endswith(".json"):  # Only JSON files
                file_path = os.path.join(category_path, file_name)
                with open(file_path, "r", encoding="utf-8") as f:
                    problem_json = json.load(f)
                data.append({
                    "file_name": file_name,
                    "problem": problem_json["problem"],
                    "level": problem_json["level"],
                    "type": problem_json["type"],
                    "solution": problem_json["solution"],
                    "label": category
                })

# Convert list of dicts to DataFrame
df = pd.DataFrame(data)
print("Sample training data:\n", df.head())

# ============================
# 2. CLEANING & PREPROCESSING TEXT
# ============================

def clean_text(text):
    """
    Clean and preprocess mathematical problem text.

    Steps:
    1. Convert to lowercase
    2. Remove special characters like $, ^, _, {, }
    3. Remove extra spaces
    """
    text = text.lower()                           # Lowercase all text
    text = re.sub(r'[$^_{}]', ' ', text)          # Remove special chars
    text = re.sub(r'\s+', ' ', text).strip()      # Remove extra whitespace
    return text

# Clean problem text
df["problem"] = df["problem"].apply(clean_text)

# Extract numeric level (if available), else set to 0
df["level"] = df["level"].str.extract(r"(\d+)").fillna(0).astype(int)

# Combine problem + level into a single feature
df["combined_text"] = df["problem"] + " Level_" + df["level"].astype(str)

# ============================
# 3. PREPARE FEATURES & TARGET
# ============================
X = df["combined_text"]  # Features (problem + level)
y = df["label"]          # Target labels

# Split into train (95%) and validation (5%)
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.05, random_state=49, stratify=y
)

# ============================
# 4. FEATURE EXTRACTION (TF-IDF)
# ============================
vectorizer = TfidfVectorizer(max_features=10000, ngram_range=(1,3),min_df=2,sublinear_tf=True)  #solution to one of the questions--appilied min_df=2 for dealing with the noise
X_train_tfidf = vectorizer.fit_transform(X_train)  # Fit and transform training data
X_val_tfidf = vectorizer.transform(X_val)          # Transform validation data

# ============================
# 5. TRAIN LOGISTIC REGRESSION
# ============================
model = LogisticRegression(max_iter=2000)
model.fit(X_train_tfidf, y_train)  # Train model on TF-IDF vectors

# ============================
# 6. EVALUATE MODEL
# ============================
y_pred = model.predict(X_val_tfidf)
y_proba = model.predict_proba(X_val_tfidf)  # Probabilities for confidence scores
print(classification_report(y_val, y_pred))  # Precision, Recall, F1

# ============================
# 7. PREDICT ON TEST SET (SORTED)
# ============================
test_dir = r"C:\Users\adity\OneDrive\Desktop\ML chalange\archive\Final\test"

# Get all test JSON filenames and sort numerically
test_files = [f for f in os.listdir(test_dir) if f.endswith(".json")]

def sort_key(file_name):
    match = re.search(r'(\d+)', file_name)  # Extract numeric part
    return int(match.group(1)) if match else 0

test_files = sorted(test_files, key=sort_key)

# Load and clean test data
test_data = []
for file_name in test_files:
    file_path = os.path.join(test_dir, file_name)
    with open(file_path, "r", encoding="utf-8") as f:
        problem_json = json.load(f)

    # Clean + preprocess test problems
    problem_text = clean_text(problem_json["problem"])
    level_value = str(re.search(r"(\d+)", str(problem_json.get("level", ""))).group(1)) if re.search(r"(\d+)", str(problem_json.get("level", ""))) else "0"
    combined_text = problem_text + " Level_" + level_value

    test_data.append({
        "file_name": file_name,
        "problem": combined_text
    })

df_test = pd.DataFrame(test_data)

# Transform test text into TF-IDF vectors
X_test_tfidf = vectorizer.transform(df_test["problem"])

# Predict categories and probabilities
test_preds = model.predict(X_test_tfidf)
test_proba = model.predict_proba(X_test_tfidf)
confidence_scores = np.max(test_proba, axis=1)  # Confidence = max predicted probability

# Prepare submission DataFrame
submission = pd.DataFrame({
    "file_name": df_test["file_name"],
    "predicted_category": test_preds,
    "confidence_score": confidence_scores
})

# ============================
# 8. SAVE SUBMISSION FILE
# ============================
submission_file = "TEAMID_CCC02A.csv"
submission.to_csv(submission_file, index=False)
print(f"Submission file saved as {submission_file}")
print(submission.head())  # Show first 5 rows
