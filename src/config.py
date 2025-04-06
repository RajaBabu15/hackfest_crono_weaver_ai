# python src/config.py

import os
from dotenv import load_dotenv

load_dotenv()

OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")
EMBEDDING_MODEL_NAME = os.environ.get("EMBEDDING_MODEL_NAME", "all-MiniLM-L6-v2")
PATHWAY_VECTOR_HOST = os.environ.get("PATHWAY_VECTOR_HOST", "localhost")
PATHWAY_VECTOR_PORT = int(os.environ.get("PATHWAY_VECTOR_PORT", "8900"))
API_PORT = int(os.environ.get("API_PORT", "8000"))
STREAMLIT_PORT = int(os.environ.get("STREAMLIT_PORT", "8501"))
INPUT_DATA_DIR = os.environ.get("INPUT_DATA_DIR", "/app/data/input")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable not set.")