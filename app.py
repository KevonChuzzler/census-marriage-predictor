from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import pandas as pd
import joblib

# 1. INITIALIZE 'app' FIRST
app = FastAPI(title="SA Inter-Ethnic Marriage Predictor API")

# 2. THEN ADD MIDDLEWARE
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 3. THEN LOAD MODELS AND DEFINE ROUTES
model = joblib.load('spouse_prediction_model.pkl')
model_columns = joblib.load('model_columns.pkl')

class UserProfile(BaseModel):
    sex: str
    age: int
    language: str
    province: str

@app.post("/predict")
def predict_probabilities(profile: UserProfile):
    person = pd.DataFrame({
        'Sex': [profile.sex],
        'Age': [profile.age],
        'Language': [profile.language],
        'Province': [profile.province]
    })
    
    person_encoded = pd.get_dummies(person)
    person_encoded = person_encoded.reindex(columns=model_columns, fill_value=0)
    
    probabilities = model.predict_proba(person_encoded)[0]
    classes = model.classes_
    
    results = {
        str(cls): float(prob * 100) 
        for cls, prob in zip(classes, probabilities)
    }
    sorted_results = dict(sorted(results.items(), key=lambda item: item[1], reverse=True))
    
    return {"predictions": sorted_results}