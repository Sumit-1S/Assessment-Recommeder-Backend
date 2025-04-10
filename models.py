# models.py
from pydantic import BaseModel
from typing import List

class Assessment(BaseModel):
    name: str
    url: str
    adaptive_support: str
    description: str
    duration: str  # keep as string to support formats like "max 60 minutes"
    remote_support: str
    test_type: List[str]

class UserQuery(BaseModel):
    query: str

class PredictionResponse(BaseModel):
    recommended_assessments: List[Assessment]
