import os
API_KEY = os.getenv("API_KEY")
BASE_URL = "https://www.shl.com"
CACHE_FILE = "app/cache/shl_gemini_embeddings.pkl"
TOP_K = 10
TOP_K_INDEXES = 30
DEFAULT_TOP_K = 5
PATH_TESTS_INFO = "shl_assessments_full.csv"
PATH_EMBEDDINGS = "shl_gemini_embeddings.pkl"
EMBEDDING_GENAI_MODEL = "embedding-001"
RESPONSE_GENERATOR_GENAI_MODEL = "gemini-2.0-flash-001"