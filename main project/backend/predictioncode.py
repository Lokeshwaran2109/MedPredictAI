
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier
import joblib

print("Loading Dataset...")
df = pd.read_csv("heart_2020_cleaned.csv")

# Convert Yes/No to 1/0
targets = ['HeartDisease', 'KidneyDisease', 'SkinCancer']
for col in targets:
    df[col] = df[col].map({'Yes':1, 'No':0})

# Label Encode categorical columns
le = LabelEncoder()
for col in df.columns:
    if df[col].dtype == 'object':
        df[col] = le.fit_transform(df[col])

# Function to train and save model with feature names
def train_model(target):
    print(f"\nTraining Model for {target}...")

    X = df.drop(targets, axis=1)
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    sm = SMOTE(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    model = XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        random_state=42
    )

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print(f"{target} Accuracy: {acc*100:.2f}%")

    # SAVE model + feature names together
    joblib.dump((model, X.columns.tolist()), f"{target}_model.pkl")
    print(f"{target}_model.pkl Saved Successfully!")

# Train all 3
train_model('HeartDisease')
train_model('KidneyDisease')
train_model('SkinCancer')