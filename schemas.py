from pydantic import BaseModel, Field
from typing import List, Dict, Optional

class PredictionRequest(BaseModel):
    manufacturer: str = Field(..., description="Bus manufacturer (Empire, MBMT, DHERADUN)")
    amax_cell_temp: float = Field(..., description="A Max Cell Temperature")
    bmax_cell_temp: float = Field(..., description="B Max Cell Temperature")
    thermistor1: float = Field(..., description="BCS Thermistor 1 Reading")
    thermistor2: float = Field(..., description="BCS Thermistor 2 Reading")

class FeatureImportance(BaseModel):
    feature: str
    importance: float

class PredictionResponse(BaseModel):
    fault_detected: bool
    probability: float
    confidence_level: str
    manufacturer: str
    timestamp: str
    features_used: List[str]
    input_data: Dict[str, float]

class ModelMetadata(BaseModel):
    name: str
    version: str
    accuracy: float
    recall: float
    features: List[str]

class HealthCheck(BaseModel):
    status: str
    available_models: List[str]
