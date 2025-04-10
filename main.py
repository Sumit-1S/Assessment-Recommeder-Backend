# app/main.py
from fastapi import FastAPI, status
from fastapi.responses import JSONResponse
from models import UserQuery, PredictionResponse, Assessment
from utils import (
    load_documents_and_embeddings,
    build_index,
    extract_url,
    fetch_description,
    safe_embed_content,
    search_index,
    generate_response,
    client,
    TOP_K_INDEXES
)
import numpy as np
import pandas as pd
import os
import json

app = FastAPI()

# Globals
documents = []
embeddings = []
index = None
catalog_df = pd.read_csv("shl_assessments_full.csv")

RECOMMENDATION_FIELDS = [
    "Assessment Name", "Test Type", "Remote Testing Support", "Adaptive/IRT Support", "URL"
]
BASE_COURSE_URL = "https://www.shl.com/solutions/products/product-catalog/"

@app.on_event("startup")
def load_resources():
    global documents, embeddings, index
    print("🔄 Loading documents and embeddings at startup...")
    documents, embeddings = load_documents_and_embeddings()
    index = build_index(embeddings)
    print(f"✅ Loaded {len(documents)} documents and built FAISS index.")

@app.post("/recommend")
async def predict_assessments(user_query: UserQuery):
    query = user_query.query

    url = extract_url(query)
    description = fetch_description(url) if url else query

    query_embedding = safe_embed_content(description)
    indices = search_index(query_embedding, index=index, k=TOP_K_INDEXES)
    context = "\n\n".join([documents[i] for i in indices])

    raw_response = generate_response(query, context)

    # Parse JSON block from LLM response (remove markdown formatting and decode)
    if isinstance(raw_response, str):
        try:
            cleaned = raw_response.strip().strip("```json").strip("```")
            parsed = json.loads(cleaned)
        except json.JSONDecodeError:
            parsed = []
    else:
        parsed = raw_response

    result = []
    for item in parsed:
        exam_names = item["Exam Name"]
        durations = item["Duration"]

        for i, exam in enumerate(exam_names):
            match = catalog_df[catalog_df["Assessment Name"].str.lower() == exam.lower()]
            if not match.empty:
                row = match.iloc[0][RECOMMENDATION_FIELDS].to_dict()
                print(row)
                result.append({
                    "name": row["Assessment Name"],
                    "url": row["URL"],
                    "adaptive_support": row["Adaptive/IRT Support"],
                    "description": exam,
                    "duration": durations[i] if i < len(durations) else "Unknown",
                    "remote_support": row["Remote Testing Support"],
                    "test_type": [row["Test Type"]] if isinstance(row["Test Type"], str) else row["Test Type"]
                })
            else:
                result.append({
                    "name": exam,
                    "url": BASE_COURSE_URL,
                    "adaptive_support": "Unknown",
                    "description": exam,
                    "duration": durations[i] if i < len(durations) else "Unknown",
                    "remote_support": "Unknown",
                    "test_type": ["Unknown"]
                })

    return JSONResponse(status_code=status.HTTP_200_OK, content={"recommended_assessments": result})


# Health Check Endpoint
@app.get("/health")
def health_check():
    return JSONResponse(status_code=status.HTTP_200_OK, content={"status": "healthy"})
