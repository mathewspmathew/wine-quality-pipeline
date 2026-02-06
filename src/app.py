"""
Wine Quality API - FastAPI Application

"""
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List
import uvicorn
from src.predict import predict
import os

app = FastAPI(
    title="Wine Quality Classifier API",
    description="ML API for wine quality classification",
    version="1.0.0"
)

class PredictionRequest(BaseModel):
    features: List[float]

class PredictionResponse(BaseModel):
    prediction: str
    features: List[float]

@app.get("/")
def home():
    return {
        "message": "Wine Quality Classifier API",
        "version": "1.0.0",
        "endpoints": {
            "health": "/health",
            "predict": "/predict",
            "docs": "/docs"
        }
    }

@app.get("/health")
def health():
    model_exists = os.path.exists('models/wine_model.pkl')
    return {
        "status": "healthy" if model_exists else "unhealthy",
        "model_loaded": model_exists
    }

@app.post("/predict", response_model=PredictionResponse)
def make_prediction(request: PredictionRequest):
    try:
        if len(request.features) != 13:
            raise HTTPException(
                status_code=400,
                detail="Expected 13 features"
            )
        
        result = predict(request.features)
        
        return PredictionResponse(
            prediction=result,
            features=request.features
        )
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)