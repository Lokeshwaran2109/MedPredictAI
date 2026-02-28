from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# -----------------------------------
# Initialize FastAPI App
# -----------------------------------
app = FastAPI()

# -----------------------------------
# CORS Configuration (DEV MODE)
# -----------------------------------
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# -----------------------------------
# Load Model
# -----------------------------------
try:
    model, feature_names = joblib.load("HeartDisease_model.pkl")
    print("✅ Model loaded successfully")
    print("📌 Model expects features:", feature_names)
except Exception as e:
    raise RuntimeError(f"❌ Model loading failed: {e}")

# -----------------------------------
# Request Schema
# -----------------------------------
class HealthInput(BaseModel):
    BMI: float
    Smoking: int
    AlcoholDrinking: int
    Stroke: int
    PhysicalHealth: float
    MentalHealth: float
    DiffWalking: int
    Sex: int
    AgeCategory: int
    Race: int   # ✅ ADD THIS
    Diabetic: int
    PhysicalActivity: int
    GenHealth: int
    SleepTime: float
    Asthma: int
# -----------------------------------
# Health Check Route
# -----------------------------------
@app.get("/")
def home():
    return {"message": "Backend running successfully 🚀"}

# -----------------------------------
# Prediction Route
# -----------------------------------
@app.post("/predict")
def predict(data: HealthInput):
    try:
        # Convert request to dictionary
        input_dict = data.dict()

        # Convert to DataFrame
        input_df = pd.DataFrame([input_dict])

        # 🔥 Ensure model feature order matches
        missing_cols = set(feature_names) - set(input_df.columns)
        if missing_cols:
            raise HTTPException(
                status_code=400,
                detail=f"Missing columns in request: {missing_cols}"
            )

        input_df = input_df[feature_names]

        # Prediction
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]

        return {
            "prediction": int(prediction),
            "probability": float(probability)
        }

    except HTTPException as e:
        raise e

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )