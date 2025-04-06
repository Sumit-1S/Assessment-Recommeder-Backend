from fastapi import FastAPI
from models import UserQuery, PredictionResponse
from utils import (
    load_documents_and_embeddings,
    build_index,
    extract_url,
    fetch_description,
    safe_embed_content,
    search_index,
    generate_response,
)
from config import *
import numpy as np

app = FastAPI()

# Globals
documents = []
embeddings = []
index = None

@app.on_event("startup")
def load_resources():
    global documents, embeddings, index
    print("🔄 Loading documents and embeddings at startup...")
    documents, embeddings = load_documents_and_embeddings()
    index = build_index(embeddings)
    print(f"✅ Loaded {len(documents)} documents and built FAISS index.")

@app.post("/recommend", response_model=PredictionResponse)
async def predict_assessments(user_query: UserQuery):
    query = user_query.description

    url = extract_url(query)
    description = fetch_description(url) if url else query

    query_embedding = safe_embed_content(description)
    indices = search_index(query_embedding, index=index, k=TOP_K_INDEXES)
    context = "\n\n".join([documents[i] for i in indices])

    response = generate_response(query, context)
    return {"recommended_assessments": response}