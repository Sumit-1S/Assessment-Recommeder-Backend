# app/utils.py
import re
import time
import pickle
import requests
import numpy as np
import pandas as pd
import faiss
from bs4 import BeautifulSoup
from google import genai
from config import *

client = genai.Client(api_key=API_KEY)

def load_documents_and_embeddings():
    df = pd.read_csv(PATH_TESTS_INFO)
    docs = []
    for _, row in df.iterrows():
        text = f"""
        Assessment Name: {row['Assessment Name']}
        Description: {row['Description']}
        Job Levels: {row['Job Levels']}
        Test Type: {row['Test Type']}
        Remote Support: {row['Remote Testing Support']}
        Adaptive Support: {row['Adaptive/IRT Support']}
        Languages: {row['Languages']}
        Duration: {row['Assessment Length']}
        """
        docs.append(text.strip())

    with open(PATH_EMBEDDINGS, "rb") as f:
        embs = pickle.load(f)
    embs = np.array(embs, dtype='float32')

    return docs, embs

def build_index(embedding_matrix):
    dimension = embedding_matrix.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embedding_matrix)
    return index

def embed_content(text):
    response = client.models.embed_content(
        model=EMBEDDING_GENAI_MODEL,
        contents=text
    )
    return response

def safe_embed_content(text, retries=3, delay=5):
    for attempt in range(retries):
        try:
            response = embed_content(text)
            return response.embeddings[0].values
        except Exception as e:
            print(f"Attempt {attempt + 1} failed: {e}")
            if attempt < retries - 1:
                time.sleep(delay)
            else:
                raise e

def search_index(query_embedding, index, k):
    query_array = np.array([query_embedding], dtype='float32')
    distances, indices = index.search(query_array, k)
    return indices[0]

def generate_response(query, content):
    response = client.models.generate_content(
        model=RESPONSE_GENERATOR_GENAI_MODEL,
        contents=[
            {
                "role": "user",
                "parts": [
                    {"text": f"""
                    You are an SHL Assessment Assistant.

                    Given the following context:
                    {content}

                    Give top {TOP_K} tests from the context that may be best suited for the following profile: {query}

                    **DO NOT ASSUME ANY TEST FROM YOUR KNOWLEDGE BASE**
                    **GIVE OUTPUT IN PROVIDED FORMAT BELOW ONLY**
                    **MAY GIVE COMBINATION OF TEST IF USER SEPCIFIES THE DURATION**
                    **DON'T GENERATE MORE THAN SPECIFIED NUMBER OF SUGGESTIONS**

                   Return a JSON array like:
                      ```json
                      [
                        {{
                          "Exam Name": ["Java Platform Enterprise Edition 7 (Java EE 7)"],
                          "Duration": ["30 minutes"]
                        }},
                        ...
                      ]
                    ```
                    """}
                ]
            }
        ],
    )
    return response.text

def extract_url(text):
    urls = re.findall(r'(https?://\S+)', text)
    return urls[0] if urls else None

def fetch_description(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, "html.parser")
        text_content = soup.get_text(separator=" ", strip=True)
        prompt = f"""
            You are given the content of a web page that may contain a job description.
            Your task is to extract only the job description from the content.
            **EXTRACT THE RESPONSIBILITIES AND REQUIREMENTS OF THE CANDIDATE ONLY**
            **DON'T extract any other INFORMATION**
            Web page content:
            \"""
            {text_content}
            \"""
            OUTPUT in the following format:
            {{
                "Extracted Job Description" :"job_description",
            }}
        """
        response = client.models.generate_content(
            model=RESPONSE_GENERATOR_GENAI_MODEL,
            contents=[
                {
                    "role": "user",
                    "parts": [
                        {"text": prompt}
                    ]
                }
            ]
        )
        job_description = response.text.strip()
        raw_json = job_description.replace("```json", "").replace("```", "").strip()
        parsed = eval(raw_json)
        return parsed["Extracted Job Description"]
    except Exception as e:
        print(f"Error in fetch_description: {e}")
        return None
