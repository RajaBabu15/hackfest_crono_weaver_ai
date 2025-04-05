import os
import csv
import numpy as np
import streamlit as st
import time
from datetime import datetime
import threading
from sentence_transformers import SentenceTransformer
from typing import List, Dict
import ast

# Configuration
MODEL_NAME = 'all-MiniLM-L6-v2'
DEFAULT_INPUT_DIR = "/app/data/input"
DEFAULT_OUTPUT_PATH = "/app/data/output/indexed_tickets.csv"

# For monitoring
last_check_time = datetime.now()
last_modification_time = None
last_file_count = 0
last_ticket_count = 0

@st.cache_resource
def load_model():
    """Load the sentence transformer model"""
    return SentenceTransformer(MODEL_NAME)

def load_embeddings(output_path) -> tuple:
    """Load embeddings with proper dimension handling"""
    embeddings = []
    tickets = []
    
    if not os.path.exists(output_path):
        st.error(f"File not found: {output_path}")
        return np.array([]), []
    
    with st.spinner(f"Loading embeddings from: {output_path}"):
        try:
            with open(output_path, 'r', encoding='utf-8-sig') as f:
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
                            st.warning(f"Row {row_idx}: Invalid shape {embedding.shape}, expected (384,)")
                            continue
                        
                        embeddings.append(embedding)
                        tickets.append(row)
                        
                    except Exception as e:
                        st.warning(f"Row {row_idx}: {str(e)}")
                        continue
                        
            return np.array(embeddings), tickets
        except Exception as e:
            st.error(f"Error loading embeddings: {str(e)}")
            return np.array([]), []

def search_tickets(query: str, model, embeddings: np.ndarray, tickets: list, top_k: int = 5) -> List[Dict]:
    """Enhanced semantic search with error handling"""
    if len(embeddings) == 0:
        return []
    
    try:
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
        st.error(f"Search error: {str(e)}")
        return []

def monitor_input_directory(input_dir):
    """Check for new files in the input directory"""
    global last_file_count
    try:
        files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
        current_count = len(files)
        
        # Sort files by modification time to show the most recent ones
        files.sort(key=lambda x: os.path.getmtime(os.path.join(input_dir, x)), reverse=True)
        
        # Get the 5 most recent files
        recent_files = files[:5]
        
        # Format the result with timestamps
        results = []
        for file in recent_files:
            file_path = os.path.join(input_dir, file)
            mod_time = datetime.fromtimestamp(os.path.getmtime(file_path))
            file_size = os.path.getsize(file_path)
            
            # Count tickets in this file
            ticket_count = 0
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    # Skip header
                    next(csv.reader(f))
                    ticket_count = sum(1 for _ in csv.reader(f))
            except Exception:
                pass
                
            results.append({
                "file": file,
                "modified": mod_time.strftime("%Y-%m-%d %H:%M:%S"),
                "size": f"{file_size/1024:.2f} KB",
                "tickets": ticket_count
            })
            
        return {
            "total_files": current_count,
            "new_files": current_count - last_file_count if current_count > last_file_count else 0,
            "recent_files": results
        }
    except Exception as e:
        return {
            "total_files": 0,
            "new_files": 0,
            "recent_files": [],
            "error": str(e)
        }

def monitor_output_file(output_path):
    """Check if the output file has been updated"""
    global last_modification_time, last_ticket_count
    
    try:
        if not os.path.exists(output_path):
            return {
                "exists": False,
                "message": "File does not exist yet",
                "ticket_count": 0,
                "updated": False
            }
        
        # Get current modification time
        current_mod_time = datetime.fromtimestamp(os.path.getmtime(output_path))
        file_size = os.path.getsize(output_path)
        
        # Count tickets in file
        ticket_count = 0
        try:
            with open(output_path, 'r', encoding='utf-8') as f:
                # Skip header
                next(csv.reader(f))
                ticket_count = sum(1 for _ in csv.reader(f))
        except Exception:
            pass
        
        # Check if file was updated
        updated = False
        if last_modification_time is None or current_mod_time > last_modification_time:
            updated = True
            last_modification_time = current_mod_time

        # Check if count changed
        new_tickets = 0
        if ticket_count > last_ticket_count:
            new_tickets = ticket_count - last_ticket_count
            last_ticket_count = ticket_count
            
        return {
            "exists": True,
            "mod_time": current_mod_time.strftime("%Y-%m-%d %H:%M:%S"),
            "size": f"{file_size/1024:.2f} KB",
            "ticket_count": ticket_count,
            "new_tickets": new_tickets,
            "updated": updated
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e),
            "updated": False
        }

def main():
    st.title("Semantic Ticket Search")
    
    # Add tabs for different features
    tab1, tab2, tab3 = st.tabs(["Search", "Monitor", "Settings"])
    
    with tab3:
        st.subheader("Settings")
        
        # Input directory selection (disabled)
        st.text("Input Data Directory (not editable)")
        input_dir = st.text_input("Input Directory", value=DEFAULT_INPUT_DIR, disabled=True, key="input_dir_setting")
        
        # Output file path with default
        output_path = st.text_input("Output CSV Path", value=DEFAULT_OUTPUT_PATH, key="output_path_setting")
        
        # Save settings to session state
        if st.button("Save Settings"):
            st.session_state['input_dir'] = input_dir
            st.session_state['output_path'] = output_path
            st.success("Settings saved!")
    
    # Initialize session state variables if not already set
    if 'input_dir' not in st.session_state:
        st.session_state['input_dir'] = DEFAULT_INPUT_DIR
    if 'output_path' not in st.session_state:
        st.session_state['output_path'] = DEFAULT_OUTPUT_PATH
    
    # Monitoring tab
    with tab2:
        st.subheader("Monitoring")
        
        # Create two columns for monitoring
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Input Directory Monitor")
            input_monitor = st.empty()
        
        with col2:
            st.markdown("### Output File Monitor")
            output_monitor = st.empty()
        
        # Auto-refresh button
        auto_refresh = st.checkbox("Auto-refresh every 5 seconds", value=False)
        
        # Manual refresh button
        refresh_button = st.button("Refresh Now")
        
        # Update the monitoring information
        if refresh_button or auto_refresh:
            # Monitor input directory
            input_stats = monitor_input_directory(st.session_state['input_dir'])
            
            # Display input directory stats
            with input_monitor.container():
                st.write(f"Total CSV files: {input_stats['total_files']}")
                if input_stats['new_files'] > 0:
                    st.success(f"🆕 {input_stats['new_files']} new files since last check!")
                
                if len(input_stats['recent_files']) > 0:
                    st.write("Recent files:")
                    for file_info in input_stats['recent_files']:
                        st.info(
                            f"📄 {file_info['file']}\n"
                            f"Modified: {file_info['modified']}\n"
                            f"Size: {file_info['size']}\n"
                            f"Tickets: {file_info['tickets']}"
                        )
                else:
                    st.write("No CSV files found in input directory.")
            
            # Update last file count
            global last_file_count
            last_file_count = input_stats['total_files']
            
            # Monitor output file
            output_stats = monitor_output_file(st.session_state['output_path'])
            
            # Display output file stats
            with output_monitor.container():
                if output_stats['exists']:
                    st.write(f"Output file: {os.path.basename(st.session_state['output_path'])}")
                    st.write(f"Last modified: {output_stats['mod_time']}")
                    st.write(f"Size: {output_stats['size']}")
                    st.write(f"Total indexed tickets: {output_stats['ticket_count']}")
                    
                    if output_stats['updated']:
                        st.success("🔄 File was updated!")
                    if output_stats.get('new_tickets', 0) > 0:
                        st.success(f"➕ {output_stats['new_tickets']} new tickets indexed!")
                else:
                    if 'error' in output_stats:
                        st.error(f"Error: {output_stats['error']}")
                    else:
                        st.warning(f"Output file does not exist: {st.session_state['output_path']}")
            
            # Auto-refresh logic
            if auto_refresh:
                time.sleep(5)
                st.experimental_rerun()
    
    # Search tab
    with tab1:
        st.subheader("Ticket Search")
        
        # Load the model
        model = load_model()
        
        # Load button
        if st.button("Load Embeddings"):
            embeddings, tickets = load_embeddings(st.session_state['output_path'])
            st.session_state['embeddings'] = embeddings
            st.session_state['tickets'] = tickets
            
            if len(tickets) > 0:
                st.success(f"Successfully loaded {len(tickets)} embeddings")
            else:
                st.error("No embeddings loaded. Please check the file path.")
        
        # Search functionality
        query = st.text_input("Enter your search query")
        top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
        
        if query and st.button("Search"):
            if 'embeddings' not in st.session_state or len(st.session_state['embeddings']) == 0:
                st.warning("Please load embeddings first")
            else:
                with st.spinner("Searching..."):
                    results = search_tickets(query, model, st.session_state['embeddings'], 
                                            st.session_state['tickets'], top_k)
                
                if not results:
                    st.info("No results found")
                else:
                    st.subheader(f"Top {len(results)} matches:")
                    for i, res in enumerate(results, 1):
                        with st.expander(f"#{i}: {res['subject']} (Score: {res['similarity']:.4f})"):
                            cols = st.columns(2)
                            with cols[0]:
                                st.write("**Ticket ID:**", res['ticket_id'])
                                st.write("**Customer:**", res['customer_id'])
                                st.write("**Timestamp:**", res['timestamp'])
                            with cols[1]:
                                st.write("**Similarity:**", f"{res['similarity']:.4f}")
                            
                            st.markdown("**Subject:**")
                            st.info(res['subject'])
                            
                            st.markdown("**Content:**")
                            st.text_area("", value=res['body'], height=150, key=f"body_{i}")

if __name__ == "__main__":
    main()













# import os
# import csv
# import numpy as np
# import streamlit as st
# from sentence_transformers import SentenceTransformer
# from typing import List, Dict
# import ast

# # Configuration
# MODEL_NAME = 'all-MiniLM-L6-v2'
# DEFAULT_INPUT_DIR = "/app/data/input"
# DEFAULT_OUTPUT_PATH = "/app/data/output/indexed_tickets.csv"

# @st.cache_resource
# def load_model():
#     """Load the sentence transformer model"""
#     return SentenceTransformer(MODEL_NAME)

# def load_embeddings(output_path) -> tuple:
#     """Load embeddings with proper dimension handling"""
#     embeddings = []
#     tickets = []
    
#     if not os.path.exists(output_path):
#         st.error(f"File not found: {output_path}")
#         return np.array([]), []
    
#     with st.spinner(f"Loading embeddings from: {output_path}"):
#         try:
#             with open(output_path, 'r', encoding='utf-8-sig') as f:
#                 reader = csv.DictReader(f)
                
#                 for row_idx, row in enumerate(reader):
#                     try:
#                         # Clean and parse embedding
#                         embedding_str = row['embedding'].strip()
                        
#                         # Remove surrounding quotes if present
#                         if embedding_str.startswith('"') and embedding_str.endswith('"'):
#                             embedding_str = embedding_str[1:-1]
                        
#                         # Convert string to list and flatten
#                         embedding_list = ast.literal_eval(embedding_str)
#                         if isinstance(embedding_list[0], list):  # Handle nested lists
#                             embedding_list = embedding_list[0]
                        
#                         embedding = np.array(embedding_list, dtype=np.float32).flatten()
                        
#                         # Verify dimensions
#                         if embedding.shape != (384,):
#                             st.warning(f"Row {row_idx}: Invalid shape {embedding.shape}, expected (384,)")
#                             continue
                        
#                         embeddings.append(embedding)
#                         tickets.append(row)
                        
#                     except Exception as e:
#                         st.warning(f"Row {row_idx}: {str(e)}")
#                         continue
                        
#             return np.array(embeddings), tickets
#         except Exception as e:
#             st.error(f"Error loading embeddings: {str(e)}")
#             return np.array([]), []

# def search_tickets(query: str, model, embeddings: np.ndarray, tickets: list, top_k: int = 5) -> List[Dict]:
#     """Enhanced semantic search with error handling"""
#     if len(embeddings) == 0:
#         return []
    
#     try:
#         query_embedding = model.encode([query], show_progress_bar=False)[0]
        
#         # Normalize embeddings
#         embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1)[:, np.newaxis]
#         query_norm = query_embedding / np.linalg.norm(query_embedding)
        
#         # Cosine similarity
#         similarities = np.dot(embeddings_norm, query_norm)
        
#         # Get top results
#         top_indices = np.argsort(similarities)[-top_k:][::-1]
#         return [{
#             **tickets[i],
#             "similarity": float(similarities[i]),
#             "combined_text": f"{tickets[i]['subject']}\n{tickets[i]['body'][:200]}..."
#         } for i in top_indices]
    
#     except Exception as e:
#         st.error(f"Search error: {str(e)}")
#         return []

# def main():
#     st.title("Semantic Ticket Search")
    
#     # Input directory selection (disabled)
#     st.text("Input Data Directory (not editable)")
#     input_dir = st.text_input("", value=DEFAULT_INPUT_DIR, disabled=True)
    
#     # Output file path with default
#     output_path = st.text_input("Output CSV Path", value=DEFAULT_OUTPUT_PATH)
    
#     # Load the model
#     model = load_model()
    
#     # Load button
#     if st.button("Load Embeddings"):
#         embeddings, tickets = load_embeddings(output_path)
#         st.session_state['embeddings'] = embeddings
#         st.session_state['tickets'] = tickets
        
#         if len(tickets) > 0:
#             st.success(f"Successfully loaded {len(tickets)} embeddings")
#         else:
#             st.error("No embeddings loaded. Please check the file path.")
    
#     # Search functionality
#     st.subheader("Search Tickets")
#     query = st.text_input("Enter your search query")
#     top_k = st.slider("Number of results", min_value=1, max_value=20, value=5)
    
#     if query and st.button("Search"):
#         if 'embeddings' not in st.session_state or len(st.session_state['embeddings']) == 0:
#             st.warning("Please load embeddings first")
#         else:
#             with st.spinner("Searching..."):
#                 results = search_tickets(query, model, st.session_state['embeddings'], 
#                                          st.session_state['tickets'], top_k)
            
#             if not results:
#                 st.info("No results found")
#             else:
#                 st.subheader(f"Top {len(results)} matches:")
#                 for i, res in enumerate(results, 1):
#                     with st.expander(f"#{i}: {res['subject']} (Score: {res['similarity']:.4f})"):
#                         cols = st.columns(2)
#                         with cols[0]:
#                             st.write("**Ticket ID:**", res['ticket_id'])
#                             st.write("**Customer:**", res['customer_id'])
#                             st.write("**Timestamp:**", res['timestamp'])
#                         with cols[1]:
#                             st.write("**Similarity:**", f"{res['similarity']:.4f}")
                        
#                         st.markdown("**Subject:**")
#                         st.info(res['subject'])
                        
#                         st.markdown("**Content:**")
#                         st.text_area("", value=res['body'], height=150, key=f"body_{i}")

# if __name__ == "__main__":
#     main()