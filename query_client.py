# query_client.py

import os
import csv
import numpy as np
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import ast

# Configuration - Windows path
MODEL_NAME = 'all-MiniLM-L6-v2'
OUTPUT_PATH = r"C:\Users\rajab\Documents\workspace\meta1\local_output_data\indexed_tickets.csv"

def load_embeddings() -> tuple:
    """Load embeddings with proper dimension handling"""
    embeddings = []
    tickets = []
    
    if not os.path.exists(OUTPUT_PATH):
        raise FileNotFoundError(f"File not found: {OUTPUT_PATH}")
    
    print(f"Loading from: {OUTPUT_PATH}")
    
    with open(OUTPUT_PATH, 'r', encoding='utf-8-sig') as f:
        reader = csv.DictReader(f)
        
        for row_idx, row in enumerate(reader):
            try:
                # Clean and parse embedding
                embedding_str = row['embedding'].strip()
                
                # Remove surrounding quotes if present
                if embedding_str.startswith('"') and embedding_str.endswith('"'):
                    embedding_str = embedding_str[1:-1]
                
                # Convert string to list and flatten
                embedding_list = ast.literal_eval(embedding_str)
                if isinstance(embedding_list[0], list):  # Handle nested lists
                    embedding_list = embedding_list[0]
                
                embedding = np.array(embedding_list, dtype=np.float32).flatten()
                
                # Verify dimensions
                if embedding.shape != (384,):
                    raise ValueError(f"Invalid shape {embedding.shape}, expected (384,)")
                
                embeddings.append(embedding)
                tickets.append(row)
                
            except Exception as e:
                print(f"Row {row_idx}: {str(e)}")
                continue
    
    print(f"Successfully loaded {len(tickets)} embeddings")
    return np.array(embeddings), tickets

def search_tickets(query: str, embeddings: np.ndarray, tickets: list, top_k: int = 5) -> List[Dict]:
    """Enhanced semantic search with error handling"""
    if len(embeddings) == 0:
        return []
    
    try:
        model = SentenceTransformer(MODEL_NAME)
        query_embedding = model.encode([query], show_progress_bar=False)[0]
        
        # Normalize embeddings
        embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
        query_norm = query_embedding / np.linalg.norm(query_embedding)
        
        # Cosine similarity
        similarities = np.dot(embeddings_norm, query_norm)
        
        # Get top results
        top_indices = np.argsort(similarities)[-top_k:][::-1]
        return [{
            **tickets[i],
            "similarity": float(similarities[i]),
            "combined_text": f"{tickets[i]['subject']}\n{tickets[i]['body'][:200]}..."
        } for i in top_indices]
    
    except Exception as e:
        print(f"Search error: {str(e)}")
        return []

if __name__ == "__main__":
    print("Initializing query client...")
    try:
        embeddings, tickets = load_embeddings()
        
        while True:
            query = input("\nEnter search query (or 'exit'): ")
            if query.lower() == 'exit':
                break
                
            results = search_tickets(query, embeddings, tickets)
            
            if not results:
                print("No results found")
                continue
                
            print(f"\nTop {len(results)} matches:")
            for i, res in enumerate(results, 1):
                print(f"\n#{i} Similarity: {res['similarity']:.4f}")
                print(f"Ticket ID: {res['ticket_id']}")
                print(f"Customer: {res['customer_id']}")
                print(f"Subject: {res['subject']}")
                print(f"Preview: {res['body'][:100]}...")
                
    except Exception as e:
        print(f"Fatal error: {str(e)}")
    finally:
        print("\nQuery client terminated.")