import os
import joblib
import pandas as pd
from datetime import datetime
from fastapi import FastAPI, HTTPException
from schemas import PredictionRequest, PredictionResponse, HealthCheck, ModelMetadata
from typing import Dict

app = FastAPI(
    title="BCS Fault Prediction API",
    description="Real-time fault prediction for Battery Cooling Systems across different bus manufacturers.",
    version="1.0.0"
)

# Global dictionary to store loaded models
models = {}

MODEL_MAPPING = {
    "Empire": "bcs_fault_model_combined.pkl",
    "MBMT": "bcs_fault_model_MH04LQ9368.pkl",
    "DHERADUN": "bcs_fault_model_dheradun.pkl"
}

@app.on_event("startup")
async def load_models():
    """Load all manufacturer models on startup."""
    print("Loading models...")
    for manufacturer, filename in MODEL_MAPPING.items():
        try:
            if os.path.exists(filename):
                models[manufacturer] = joblib.load(filename)
                print(f"Loaded {manufacturer} model from {filename}")
            else:
                print(f"Warning: {filename} not found for {manufacturer}")
        except Exception as e:
            print(f"Error loading {manufacturer} model: {e}")

@app.get("/", response_model=HealthCheck)
async def health_check():
    """Return API health status and available models."""
    return {
        "status": "healthy",
        "available_models": list(models.keys())
    }

@app.post("/predict", response_model=PredictionResponse)
async def predict_fault(request: PredictionRequest):
    """Make a fault prediction based on sensor data."""
    if request.manufacturer not in models:
        raise HTTPException(
            status_code=404, 
            detail=f"Model for manufacturer '{request.manufacturer}' not found or not loaded."
        )
    
    # Prepare input data
    input_df = pd.DataFrame([{
        "AMaxCellTemp": request.amax_cell_temp,
        "BMaxCellTemp": request.bmax_cell_temp,
        "BCSThermistor1": request.thermistor1,
        "BCSThermistor2": request.thermistor2
    }])
    
    model = models[request.manufacturer]
    
    try:
        # Get prediction and probabilities
        prediction = model.predict(input_df)[0]
        probabilities = model.predict_proba(input_df)[0]
        fault_probability = float(probabilities[1])
        
        # Determine confidence level
        if fault_probability > 0.8:
            confidence = "High"
        elif fault_probability > 0.5:
            confidence = "Medium"
        elif fault_probability > 0.3:
            confidence = "Low (Warning)"
        else:
            confidence = "Normal Operating Range"
            
        return {
            "fault_detected": bool(prediction),
            "probability": fault_probability,
            "confidence_level": confidence,
            "manufacturer": request.manufacturer,
            "timestamp": datetime.now().isoformat(),
            "features_used": input_df.columns.tolist(),
            "input_data": request.dict(exclude={"manufacturer"})
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
