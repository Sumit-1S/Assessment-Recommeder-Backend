# app/models.py
from pydantic import BaseModel

class UserQuery(BaseModel):
    description: str

class PredictionResponse(BaseModel):
    recommended_assessments: str
