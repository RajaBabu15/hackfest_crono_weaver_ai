# python src/rag.py

import llama_index
# Removed ServiceContext, Settings might be needed if choosing global config
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Document, StorageContext, Settings
from llama_index.core.vector_stores import SimpleVectorStore
from llama_index.llms.openai import OpenAI
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.core.chat_engine.condense_plus_context import CondensePlusContextChatEngine
from llama_index.core.chat_engine.types import BaseChatEngine
import logging
import pandas as pd
import numpy as np
import ast
import os
import time

from src import config

INDEXED_CSV_PATH = "/app/data/output/indexed_data.csv"

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_chat_engine_instance = None
_index_instance = None
_last_load_time = 0
_load_lock = False

# --- Create the embed model and LLM instances once ---
try:
    logger.info(f"Loading embedding model for LlamaIndex: {config.EMBEDDING_MODEL_NAME}")
    embed_model = HuggingFaceEmbedding(model_name=config.EMBEDDING_MODEL_NAME)
    logger.info("Loading LLM...")
    llm = OpenAI(model="gpt-3.5-turbo", api_key=config.OPENAI_API_KEY)
    # --- Option 1: Configure Globally (uncomment if preferred) ---
    # Settings.llm = llm
    # Settings.embed_model = embed_model
    # logger.info("LlamaIndex Settings configured globally.")
    # --- End Option 1 ---
except Exception as e:
    logger.error(f"Failed to load LLM or HuggingFace embedding model: {e}", exc_info=True)
    embed_model = None
    llm = None
# ---------------------------------------------------

def load_data_and_build_index():
    global _index_instance, _last_load_time, _load_lock

    if _load_lock: return _index_instance # Simplified return

    _load_lock = True
    logger.info(f"Attempting to load data from: {INDEXED_CSV_PATH}")

    # --- Create default empty index first ---
    if _index_instance is None:
         # Pass embed_model directly if creating empty index
         _index_instance = VectorStoreIndex([], embed_model=embed_model)
         logger.info("Initialized with empty index.")
    # -----------------------------------------

    if not os.path.exists(INDEXED_CSV_PATH):
        logger.warning(f"Index file not found: {INDEXED_CSV_PATH}. Using existing/empty index.")
        _load_lock = False
        return _index_instance

    try: current_mod_time = os.path.getmtime(INDEXED_CSV_PATH)
    except OSError as e:
         logger.error(f"Error getting modification time for {INDEXED_CSV_PATH}: {e}. Skipping load.")
         _load_lock = False
         return _index_instance

    if current_mod_time <= _last_load_time:
        logger.info("Index file has not changed. Using cached index.")
        _load_lock = False
        return _index_instance

    logger.info("Index file changed. Rebuilding index...")
    try:
        try:
            df = pd.read_csv(INDEXED_CSV_PATH)
            if df.empty: logger.warning(f"Index file {INDEXED_CSV_PATH} contains no data rows.")
            else: logger.info(f"Loaded {len(df)} rows from CSV.")
        except pd.errors.EmptyDataError:
            logger.warning(f"Index file {INDEXED_CSV_PATH} is empty.")
            df = pd.DataFrame()

        documents = []
        if not df.empty:
            required_columns = ['embedding_str', 'subject', 'body', 'ticket_id', 'timestamp', 'customer_id']
            if not all(col in df.columns for col in required_columns):
                logger.error(f"CSV file {INDEXED_CSV_PATH} missing required columns. Found: {list(df.columns)}. Required: {required_columns}")
                df = pd.DataFrame()
            else:
                for _, row in df.iterrows():
                    try:
                        embedding_list = ast.literal_eval(row['embedding_str'])
                        embedding_vector = np.array(embedding_list, dtype=np.float32)
                        if embedding_vector.shape[0] != 384:
                             logger.warning(f"Row {row.get('ticket_id', 'N/A')} has incorrect embedding dimension {embedding_vector.shape}, expected 384. Skipping.")
                             continue
                        text_content = f"Subject: {row['subject']}\nBody: {row['body']}"
                        metadata = {"ticket_id": row['ticket_id'], "timestamp": row['timestamp'], "customer_id": row['customer_id']}
                        # --- Pass embed_model=None as embedding is pre-calculated ---
                        # Note: Check LlamaIndex docs if this exact syntax changes for pre-embedded Documents
                        doc = Document(text=text_content, metadata=metadata, embedding=embedding_vector)
                        # -----------------------------------------------------------
                        documents.append(doc)
                    except Exception as parse_error:
                        logger.error(f"Error parsing row: {row.to_dict()}. Error: {parse_error}", exc_info=False)
                        continue

        if not documents:
             logger.warning("No valid documents created from CSV. Using existing/empty index.")
             # Don't overwrite existing index if parsing fails but index exists
             if _index_instance is None: _index_instance = VectorStoreIndex([], embed_model=embed_model)
        else:
             if embed_model is None:
                  raise RuntimeError("Cannot build index, embedding model failed to load.")

             logger.info(f"Building in-memory VectorStoreIndex with {len(documents)} documents.")
             storage_context = StorageContext.from_defaults(vector_store=SimpleVectorStore())
             # --- Pass embed_model directly to VectorStoreIndex ---
             # This tells the index how to interpret/compare the pre-computed embeddings if needed,
             # and which model to use if it ever needed to embed something itself (less likely here)
             _index_instance = VectorStoreIndex(
                 nodes=documents,
                 storage_context=storage_context,
                 embed_model=embed_model # Pass embed_model directly
             )
             # ---------------------------------------------------
             logger.info("In-memory index built successfully.")

        _last_load_time = current_mod_time

    except Exception as e:
        logger.error(f"Failed during index build process: {e}", exc_info=True)
        # Keep old index if loading fails
        if _index_instance is None:
            _index_instance = VectorStoreIndex([], embed_model=embed_model)
    finally:
        _load_lock = False

    return _index_instance


def get_chat_engine() -> BaseChatEngine:
    global _chat_engine_instance
    if _chat_engine_instance is None:
        try:
            # --- Ensure models are loaded ---
            if embed_model is None or llm is None:
                raise RuntimeError("Cannot create chat engine, LLM or embedding model failed to load.")
            # ---------------------------------

            logger.info("Initializing RAG components...")
            index = load_data_and_build_index()

            if index is None:
                 raise RuntimeError("Index could not be loaded or built.")

            # --- Pass embed_model directly to retriever ---
            # Retriever needs it to embed the user's query
            retriever = index.as_retriever(
                similarity_top_k=3,
                embed_model=embed_model # Pass embed_model directly
            )
            # --------------------------------------------

            logger.info("Building CondensePlusContextChatEngine")
            # --- Pass llm directly to chat engine ---
            _chat_engine_instance = CondensePlusContextChatEngine.from_defaults(
                retriever=retriever,
                llm=llm, # Pass llm directly
                system_prompt="You are a helpful assistant answering questions based on the provided context information (loaded from a file). If the context doesn't contain the answer, state that clearly.",
                verbose=False
            )
            # --------------------------------------
            logger.info("Chat engine initialized successfully.")
        except Exception as e:
            logger.error(f"Failed to initialize chat engine: {e}", exc_info=True)
            raise RuntimeError(f"Failed to initialize chat engine: {e}") from e
    return _chat_engine_instance

# Optional: Function to trigger index reload
def reload_index():
    global _last_load_time, _chat_engine_instance
    logger.info("Manual index reload triggered.")
    _last_load_time = 0
    _chat_engine_instance = None # Force rebuild of chat engine on next call
    # Rebuilding index here might be redundant if get_chat_engine always calls load_data_and_build_index
    # Consider if just clearing _chat_engine_instance is enough
    return load_data_and_build_index() # Or just let get_chat_engine handle it