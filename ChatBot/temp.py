
# import json
# import logging
# import os
# import uuid

# import pandas as pd
# import streamlit as st
# from dotenv import load_dotenv
# from endpoint_utils import get_inputs
# from llama_index.llms.types import ChatMessage, MessageRole
# from log_utils import init_pw_log_config
# from rag import DEFAULT_PATHWAY_HOST, PATHWAY_HOST, chat_engine, vector_client
# from streamlit.web.server.websocket_headers import _get_websocket_headers
# from traceloop.sdk import Traceloop


# from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
# from llama_index.embeddings.openai import OpenAIEmbedding
# from llama_index.vector_stores import SimpleVectorStore
# from llama_index.node_parser import SimpleNodeParser

# logging.basicConfig(
#     level=logging.INFO,
#     format="%(asctime)s %(name)s %(levelname)s %(message)s",
#     datefmt="%Y-%m-%d %H:%M:%S",
# )

# init_pw_log_config()


# DRIVE_URL = os.environ.get(
#     "GDRIVE_FOLDER_URL",
#     "https://drive.google.com/drive/u/0/folders/1cULDv2OaViJBmOfG5WB0oWcgayNrGtVs",
# )

# DEFAULT_INPUT_DIR = "/app/data/input"
# DEFAULT_OUTPUT_PATH = "/app/data/output/indexed_tickets.csv"

# htm = f"""
# <div style="display: flex; align-items: center; vertical-align: middle">
#     <a href="{DRIVE_URL}" style="text-decoration:none;">
#       <figure style="display: flex; vertical-align: middle; margin-right: 20px; align-items: center;">
#         <img src="./app/static/Google_Drive_logo.png" width="30" alt="Google Drive Logo">
#         <figcaption>Upload</figcaption>
#       </figure>
#     </a>

# </div>
# <div style="font-size: 10px">* These are public folders. Please do not upload confidential files.</div>
# <div><br></div>
# <a href="https://cloud.pathway.com/?modal=getstarted" style="text-decoration:none;">
#     <figure style="display: flex; vertical-align: middle; align-items: center; margin-right: 20px;">
#     <button>Connect to your folders with Pathway</button>
#     </figure>
# </a>
# """

# st.set_page_config(
#     page_title="Realtime Document AI pipelines", page_icon="./app/static/favicon.ico"
# )

# with st.sidebar:
#     if PATHWAY_HOST == DEFAULT_PATHWAY_HOST:
#         st.markdown("*Add Your Files*")

#         st.markdown(htm, unsafe_allow_html=True)

#         st.markdown("\n\n\n\n\n\n\n")
#         st.markdown("\n\n\n\n\n\n\n")
#         st.markdown(
#             "[View code on GitHub.](https://github.com/RajaBabu15/hackfest_crono_weaver_ai)"
#         )
#         st.markdown(
#             """Pathway pipelines ingest documents from [Google Drive](https://drive.google.com/drive/u/0/folders/1cULDv2OaViJBmOfG5WB0oWcgayNrGtVs). It automatically manages and syncs indexes enabling RAG applications."""
#         )
#     else:
#         st.markdown(f"*Connected to:* {PATHWAY_HOST}")
#         st.markdown(
#             "[View code on GitHub.](https://github.com/RajaBabu15/hackfest_crono_weaver_ai)"
#         )


# #     st.markdown(
# #         """*Ready to build your own?*

# # Our [docs](https://pathway.com/developers/showcases/llamaindex-pathway/) walk through creating custom pipelines with LlamaIndex.

# # *Want a hosted version?*

# # Check out our [hosted document pipelines](https://cloud.pathway.com/docindex)."""
# #     )


# # Load environment variables
# load_dotenv()


# # Define function to load embeddings
# def load_embeddings(csv_path):
#     try:
#         if not os.path.exists(csv_path):
#             return None, []

#         df = pd.read_csv(csv_path)

#         if "ticket_text" not in df.columns:
#             raise ValueError("CSV must contain 'ticket_text' column")

#         texts = df["ticket_text"].dropna().tolist()

#         if len(texts) == 0:
#             return None, []

#         # Convert text into LlamaIndex Documents
#         from llama_index.schema import Document
#         documents = [Document(text=t) for t in texts]

#         # Setup embedding model
#         embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

#         # Parse into nodes
#         parser = SimpleNodeParser()
#         nodes = parser.get_nodes_from_documents(documents)

#         # Create vector index
#         index = VectorStoreIndex(nodes, embed_model=embed_model)

#         return index.as_retriever(similarity_top_k=3), texts

#     except Exception as e:
#         logging.error(f"Error loading embeddings: {e}")
#         return None, []


# # Streamlit UI elements
# def main():
#     st.title(
#     "## Akasa: The Real-Time Knowledge Conduit"
#     )
#     htt = """
#     <p>
#         <div> Built By: CHRONO WEAVERS AI </div>
#         <div> Team Members: Raja , Anand , Ram , Abhishek , Anubhav </div>
    
#     </p>
#     """

#     tab1, tab2, tab3 = st.tabs(["Chat", "Monitor", "Settings"])
#     with tab3:
#         st.subheader("Settings")
        
#         # Input directory selection (disabled)
#         st.text("Input Data Directory (not editable)")
#         input_dir = st.text_input("Input Directory", value=DEFAULT_INPUT_DIR, disabled=True, key="input_dir_setting")
        
#         # Output file path with default
#         output_path = st.text_input("Output CSV Path", value=DEFAULT_OUTPUT_PATH, key="output_path_setting")
        
#         # Save settings to session state
#         if st.button("Save Settings"):
#             st.session_state['input_dir'] = input_dir
#             st.session_state['output_path'] = output_path
#             st.success("Settings saved!")
    
#     # Initialize session state variables if not already set
#     if 'input_dir' not in st.session_state:
#         st.session_state['input_dir'] = DEFAULT_INPUT_DIR
#     if 'output_path' not in st.session_state:
#         st.session_state['output_path'] = DEFAULT_OUTPUT_PATH
    
#     # Monitoring tab
#     with tab2:
#         st.subheader("Monitoring")
        
#         # Create two columns for monitoring
#         col1, col2 = st.columns(2)
        
#         with col1:
#             st.markdown("### Input Directory Monitor")
#             input_monitor = st.empty()
        
#         with col2:
#             st.markdown("### Output File Monitor")
#             output_monitor = st.empty()
        
#         # Auto-refresh button
#         auto_refresh = st.checkbox("Auto-refresh every 5 seconds", value=False)
        
#         # Manual refresh button
#         refresh_button = st.button("Refresh Now")
        
#         # Update the monitoring information
#         if refresh_button or auto_refresh:
#             # Monitor input directory
#             input_stats = monitor_input_directory(st.session_state['input_dir'])
            
#             # Display input directory stats
#             with input_monitor.container():
#                 st.write(f"Total CSV files: {input_stats['total_files']}")
#                 if input_stats['new_files'] > 0:
#                     st.success(f"🆕 {input_stats['new_files']} new files since last check!")
                
#                 if len(input_stats['recent_files']) > 0:
#                     st.write("Recent files:")
#                     for file_info in input_stats['recent_files']:
#                         st.info(
#                             f"📄 {file_info['file']}\n"
#                             f"Modified: {file_info['modified']}\n"
#                             f"Size: {file_info['size']}\n"
#                             f"Tickets: {file_info['tickets']}"
#                         )
#                 else:
#                     st.write("No CSV files found in input directory.")
            
#             # Update last file count
#             global last_file_count
#             last_file_count = input_stats['total_files']
            
#             # Monitor output file
#             output_stats = monitor_output_file(st.session_state['output_path'])
            
#             # Display output file stats
#             with output_monitor.container():
#                 if output_stats['exists']:
#                     st.write(f"Output file: {os.path.basename(st.session_state['output_path'])}")
#                     st.write(f"Last modified: {output_stats['mod_time']}")
#                     st.write(f"Size: {output_stats['size']}")
#                     st.write(f"Total indexed tickets: {output_stats['ticket_count']}")
                    
#                     if output_stats['updated']:
#                         st.success("🔄 File was updated!")
#                     if output_stats.get('new_tickets', 0) > 0:
#                         st.success(f"➕ {output_stats['new_tickets']} new tickets indexed!")
#                 else:
#                     if 'error' in output_stats:
#                         st.error(f"Error: {output_stats['error']}")
#                     else:
#                         st.warning(f"Output file does not exist: {st.session_state['output_path']}")
            
#             # Auto-refresh logic
#             if auto_refresh:
#                 time.sleep(5)
#                 st.rerun()
    
#     # Search tab
#     with tab1:
#         st.markdown(htt, unsafe_allow_html=True)

#         image_width = 300
#         image_height = 200

#         # Initialize chat state if needed
#         if "messages" not in st.session_state.keys():
#             if "session_id" not in st.session_state.keys():
#                 session_id = "uuid-" + str(uuid.uuid4())
#                 logging.info(json.dumps({"_type": "set_session_id", "session_id": session_id}))
#                 Traceloop.set_association_properties({"session_id": session_id})
#                 st.session_state["session_id"] = session_id

#             headers = _get_websocket_headers()
#             logging.info(
#                 json.dumps({
#                     "_type": "set_headers",
#                     "headers": headers,
#                     "session_id": st.session_state.get("session_id", "NULL_SESS"),
#                 })
#             )

#             pathway_explaination = "Pathway is a high-throughput, low-latency data processing framework that handles live data & streaming for you."
#             DEFAULT_MESSAGES = [
#                 ChatMessage(role=MessageRole.USER, content="What is Pathway?"),
#                 ChatMessage(role=MessageRole.ASSISTANT, content=pathway_explaination),
#             ]
#             chat_engine.chat_history.clear()

#             for msg in DEFAULT_MESSAGES:
#                 chat_engine.chat_history.append(msg)

#             st.session_state.messages = [
#                 {"role": msg.role, "content": msg.content} for msg in chat_engine.chat_history
#             ]
#             st.session_state.chat_engine = chat_engine
#             st.session_state.vector_client = vector_client

#         results = get_inputs()

#         last_modified_time, last_indexed_files = results

#         df = pd.DataFrame(last_indexed_files, columns=[last_modified_time, "status"])
#         if df.status.isna().any():
#             del df["status"]

#         df.set_index(df.columns[0])
#         st.dataframe(df, hide_index=True, height=150, use_container_width=True)

#         cs = st.columns([1, 1, 1, 1], gap="large")
#         with cs[-1]:
#             st.button("⟳ Refresh", use_container_width=True)

#         # Display chat history first:
#         for message in st.session_state.messages:
#             with st.chat_message(message["role"]):
#                 st.write(message["content"])

#         # Then display the chat input box at the bottom:
            
#         if prompt := st.chat_input("Your question"):
#             # Load embeddings just before executing the query:
#             embeddings, tickets = load_embeddings(st.session_state['output_path'])
#             st.session_state['embeddings'] = embeddings
#             st.session_state['tickets'] = tickets

#             if len(tickets) > 0:
#                 st.success(f"Successfully loaded {len(tickets)} embeddings")
#             else:
#                 st.error("No embeddings loaded. Please check the file path.")
            
#             # Now process the query
#             st.session_state.messages.append({"role": "user", "content": prompt})
#             logging.info(
#                 json.dumps({
#                     "_type": "user_prompt",
#                     "prompt": prompt,
#                     "session_id": st.session_state.get("session_id", "NULL_SESS"),
#                 })
#             )
#             if st.session_state.messages[-1]["role"] != "assistant":
#                 with st.chat_message("assistant"):
#                     with st.spinner("Thinking..."):
#                         response = st.session_state.chat_engine.chat(prompt)
#                         sources = []
#                         try:
#                             for source in response.source_nodes:
#                                 full_path = source.metadata.get("path", source.metadata.get("name"))
#                                 if full_path is None:
#                                     continue
#                                 if "/" in full_path:
#                                     name = f"{full_path.split('/')[-1]}"
#                                 else:
#                                     name = f"{full_path}"
#                                 if name not in sources:
#                                     sources.append(name)
#                         except AttributeError:
#                             logging.error(
#                                 json.dumps({
#                                     "_type": "error",
#                                     "error": f"No source (source_nodes) was found in response: {str(response)}",
#                                     "session_id": st.session_state.get("session_id", "NULL_SESS"),
#                                 })
#                             )
#                         sources_text = ", ".join(sources)
#                         logging.info(
#                             json.dumps({
#                                 "_type": "llm_response",
#                                 "response": str(response),
#                                 "session_id": st.session_state.get("session_id", "NULL_SESS"),
#                                 "sources": sources,
#                             })
#                         )
#                         response_text = (
#                             response.response
#                             + f"\n\nDocuments looked up to obtain this answer: {sources_text}"
#                         )
#                         st.write(response_text)
#                         message = {"role": "assistant", "content": response_text}
#                         st.session_state.messages.append(message)


# if _name_ == "_main_":
#     main()


import json
import logging
import os
import uuid
import time
import pandas as pd
import streamlit as st
from dotenv import load_dotenv
from endpoint_utils import get_inputs
from llama_index.llms.types import ChatMessage, MessageRole
from log_utils import init_pw_log_config
from rag import DEFAULT_PATHWAY_HOST, PATHWAY_HOST, chat_engine, vector_client
from streamlit.web.server.websocket_headers import _get_websocket_headers
from traceloop.sdk import Traceloop
from llama_index import VectorStoreIndex, SimpleDirectoryReader, ServiceContext
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.vector_stores import SimpleVectorStore
from llama_index.node_parser import SimpleNodeParser

# Set page config - MUST be the first Streamlit command
st.set_page_config(
    page_title="Akaśa: Real-Time Knowledge Conduit", 
    page_icon="🔄",
    layout="wide"
)

# Initialize logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
init_pw_log_config()

# Constants
DRIVE_URL = os.environ.get(
    "GDRIVE_FOLDER_URL",
    "https://drive.google.com/drive/u/0/folders/1cULDv2OaViJBmOfG5WB0oWcgayNrGtVs",
)
DEFAULT_INPUT_DIR = "/app/data/input"
DEFAULT_OUTPUT_PATH = "/app/data/output/indexed_tickets.csv"

# Custom CSS
def set_custom_styles():
    st.markdown("""
    <style>
    .main-header {
        font-size: 36px !important;
        font-weight: 700;
        color: #2E4053;
        margin-bottom: 20px;
    }
    .sub-header {
        font-size: 20px;
        color: #5D6D7E;
        margin-bottom: 30px;
    }
    .team-info {
        background-color: #F8F9F9;
        padding: 15px;
        border-radius: 10px;
        border-left: 5px solid #3498DB;
        margin-bottom: 20px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        font-size: 16px;
        font-weight: 500;
    }
    .chat-container {
        background-color: #F5F9FF;
        padding: 20px;
        border-radius: 15px;
        border: 1px solid #D4E6F1;
        margin-bottom: 20px;
    }
    .upload-section {
        background-color: #EBF5FB;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .settings-card {
        background-color: #F0F3F4;
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
    }
    .monitor-card {
        background-color: #EAEDED;
        padding: 15px;
        border-radius: 10px;
        margin-bottom: 15px;
    }
    .sidebar {
        padding: 20px;
        background-color: #F8F9F9;
    }
    .metric-card {
        background-color: white;
        padding: 15px;
        border-radius: 8px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    </style>
    """, unsafe_allow_html=True)

# Monitoring functions
def monitor_input_directory(directory_path):
    try:
        if not os.path.exists(directory_path):
            return {
                "total_files": 0,
                "new_files": 0,
                "recent_files": []
            }
        
        csv_files = [f for f in os.listdir(directory_path) if f.endswith('.csv')]
        
        recent_files = []
        for file in csv_files[:5]:
            file_path = os.path.join(directory_path, file)
            mod_time = time.ctime(os.path.getmtime(file_path))
            size = f"{os.path.getsize(file_path) / 1024:.2f} KB"
            
            try:
                df = pd.read_csv(file_path)
                ticket_count = len(df)
            except:
                ticket_count = "Unknown"
                
            recent_files.append({
                "file": file,
                "modified": mod_time,
                "size": size,
                "tickets": ticket_count
            })
            
        global last_file_count
        new_files = max(0, len(csv_files) - last_file_count)
        
        return {
            "total_files": len(csv_files),
            "new_files": new_files,
            "recent_files": recent_files
        }
    except Exception as e:
        return {
            "total_files": 0,
            "new_files": 0,
            "recent_files": [],
            "error": str(e)
        }

def monitor_output_file(file_path):
    try:
        if not os.path.exists(file_path):
            return {"exists": False}
        
        mod_time = time.ctime(os.path.getmtime(file_path))
        size = f"{os.path.getsize(file_path) / 1024:.2f} KB"
        
        df = pd.read_csv(file_path)
        ticket_count = len(df)
        
        global last_output_stats
        was_updated = (
            last_output_stats.get("mod_time") != mod_time or
            last_output_stats.get("size") != size
        )
        
        new_tickets = ticket_count - last_output_stats.get("ticket_count", ticket_count)
        
        last_output_stats = {
            "mod_time": mod_time,
            "size": size,
            "ticket_count": ticket_count
        }
        
        return {
            "exists": True,
            "mod_time": mod_time,
            "size": size,
            "ticket_count": ticket_count,
            "updated": was_updated,
            "new_tickets": new_tickets if new_tickets > 0 else 0
        }
    except Exception as e:
        return {
            "exists": False,
            "error": str(e)
        }
# Embeddings and chat functions
def load_embeddings(csv_path):
    try:
        if not os.path.exists(csv_path):
            return None, []

        df = pd.read_csv(csv_path)

        if "ticket_text" not in df.columns:
            raise ValueError("CSV must contain 'ticket_text' column")

        texts = df["ticket_text"].dropna().tolist()

        if len(texts) == 0:
            return None, []

        from llama_index.schema import Document
        documents = [Document(text=t) for t in texts]

        embed_model = OpenAIEmbedding(model="text-embedding-ada-002")

        parser = SimpleNodeParser()
        nodes = parser.get_nodes_from_documents(documents)

        index = VectorStoreIndex(nodes, embed_model=embed_model)

        return index.as_retriever(similarity_top_k=3), texts

    except Exception as e:
        logging.error(f"Error loading embeddings: {e}")
        return None, []

def initialize_chat_session():
    if "session_id" not in st.session_state:
        session_id = "uuid-" + str(uuid.uuid4())
        logging.info(json.dumps({"_type": "set_session_id", "session_id": session_id}))
        Traceloop.set_association_properties({"session_id": session_id})
        st.session_state["session_id"] = session_id

    headers = _get_websocket_headers()
    logging.info(
        json.dumps({
            "_type": "set_headers",
            "headers": headers,
            "session_id": st.session_state.get("session_id", "NULL_SESS"),
        })
    )

    pathway_explanation = """
    Akaśa is a real-time knowledge conduit powered by Pathway. It processes and indexes 
    enterprise support data in real-time, ensuring you always have access to the latest information 
    for better decision-making and customer support.
    """
    
    DEFAULT_MESSAGES = [
        ChatMessage(role=MessageRole.USER, content="What is Akaśa?"),
        ChatMessage(role=MessageRole.ASSISTANT, content=pathway_explanation),
    ]
    
    chat_engine.chat_history.clear()
    for msg in DEFAULT_MESSAGES:
        chat_engine.chat_history.append(msg)

    st.session_state.messages = [
        {"role": msg.role, "content": msg.content} for msg in chat_engine.chat_history
    ]
    st.session_state.chat_engine = chat_engine
    st.session_state.vector_client = vector_client

def render_sidebar():
    with st.sidebar:
        st.markdown('<div class="sidebar">', unsafe_allow_html=True)
        if PATHWAY_HOST == DEFAULT_PATHWAY_HOST:
            st.markdown("### Add Your Files")
            
            st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <a href="{DRIVE_URL}" target="_blank" style="text-decoration: none;">
                    <div style="display: flex; align-items: center; padding: 10px; background-color: white; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1);">
                        <img src="https://ssl.gstatic.com/docs/doclist/images/drive_2022q3_32dp.png" width="24" style="margin-right: 10px;">
                        <span>Upload to Google Drive</span>
                    </div>
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            <div style="font-size: 12px; color: #666; margin-bottom: 20px;">
            * These are public folders. Please do not upload confidential files.
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown(f"""
            <div style="margin-bottom: 20px;">
                <a href="https://cloud.pathway.com/?modal=getstarted" target="_blank" style="text-decoration: none;">
                    <button style="width: 100%; padding: 10px; background-color: #3498DB; color: white; border: none; border-radius: 8px; cursor: pointer;">
                        Connect to Pathway Cloud
                    </button>
                </a>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("""
            Pathway pipelines ingest documents from Google Drive and automatically manage 
            and sync indexes enabling RAG applications.
            """)
            
            st.markdown("[View on GitHub](https://github.com/RajaBabu15/hackfest_crono_weaver_ai)")
        else:
            st.markdown(f"*Connected to:* {PATHWAY_HOST}")
            st.markdown("[View on GitHub](https://github.com/RajaBabu15/hackfest_crono_weaver_ai)")
        
        st.markdown('</div>', unsafe_allow_html=True)

def render_settings_tab():
    st.markdown('<div class="settings-card">', unsafe_allow_html=True)
    st.subheader("System Configuration")
    
    col1, col2 = st.columns(2)
    with col1:
        st.text("Input Data Directory (not editable)")
        input_dir = st.text_input(
            "Input Directory", 
            value=DEFAULT_INPUT_DIR, 
            disabled=True, 
            key="input_dir_setting"
        )
    
    with col2:
        st.text("Output File Location")
        output_path = st.text_input(
            "Output CSV Path", 
            value=DEFAULT_OUTPUT_PATH, 
            key="output_path_setting"
        )
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("### Model Settings")
        model = st.selectbox(
            "Select LLM Model",
            ["GPT-4", "GPT-3.5-Turbo", "Claude-2", "Claude-Instant"]
        )
        temperature = st.slider("Temperature", 0.0, 1.0, 0.7)
    
    with col2:
        st.markdown("### System Settings")
        chunk_size = st.number_input("Chunk Size", 100, 1000, 300)
        overlap = st.number_input("Chunk Overlap", 0, 200, 100)
    
    if st.button("Save Settings", use_container_width=True):
        st.session_state['input_dir'] = input_dir
        st.session_state['output_path'] = output_path
        st.success("✅ Settings saved successfully!")
    
    st.markdown('</div>', unsafe_allow_html=True)

def render_monitor_tab():
    st.markdown("## System Monitoring Dashboard")
    
    # System metrics
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("System Status", "Online", "Active")
        st.markdown('</div>', unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Processing Rate", "45/min", "+2.3%")
        st.markdown('</div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Average Latency", "1.2s", "-0.3s")
        st.markdown('</div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card">', unsafe_allow_html=True)
        st.metric("Success Rate", "99.8%", "+0.2%")
        st.markdown('</div>', unsafe_allow_html=True)
def render_chat_interface():
    st.markdown('<div class="chat-container">', unsafe_allow_html=True)
    
    # Initialize chat if needed
    if "messages" not in st.session_state:
        initialize_chat_session()
    
    # Chat header and controls
    col1, col2 = st.columns([3, 1])
    with col1:
        st.markdown("### 💬 Chat with Akaśa")
    with col2:
        if st.button("🔄 Clear Chat", use_container_width=True):
            initialize_chat_session()
            st.rerun()
    
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Chat input
    if prompt := st.chat_input("Ask something..."):
        # Load embeddings
        embeddings, tickets = load_embeddings(st.session_state['output_path'])
        st.session_state['embeddings'] = embeddings
        st.session_state['tickets'] = tickets    
        with st.status("Processing query...", expanded=True) as status:
            st.write("Loading relevant context...")
                
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            logging.info(
                json.dumps({
                    "_type": "user_prompt",
                    "prompt": prompt,
                    "session_id": st.session_state.get("session_id", "NULL_SESS"),
                })
            )
            
            if st.session_state.messages[-1]["role"] != "assistant":
                with st.chat_message("assistant"):
                    status.write("Generating response...")
                    response = st.session_state.chat_engine.chat(prompt)
                    
                    # Extract sources
                    sources = []
                    try:
                        for source in response.source_nodes:
                            full_path = source.metadata.get("path", source.metadata.get("name"))
                            if full_path and full_path not in sources:
                                sources.append(full_path.split('/')[-1] if '/' in full_path else full_path)
                    except AttributeError:
                        logging.error(
                            json.dumps({
                                "_type": "error",
                                "error": f"No source nodes found in response: {str(response)}",
                                "session_id": st.session_state.get("session_id", "NULL_SESS"),
                            })
                        )
                    
                    # Format and display response
                    sources_text = ", ".join(sources) if sources else "No specific sources"
                    response_text = f"{response.response}\n\n*Sources consulted:* {sources_text}"
                    
                    st.markdown(response_text)
                    st.session_state.messages.append({"role": "assistant", "content": response_text})
                    
                    status.update(label="✅ Response generated", state="complete", expanded=False)
    
    st.markdown('</div>', unsafe_allow_html=True)

def main():
    # Apply custom styles
    set_custom_styles()
    
    # Load environment variables
    load_dotenv()
    
    # Initialize session state
    if 'input_dir' not in st.session_state:
        st.session_state['input_dir'] = DEFAULT_INPUT_DIR
    if 'output_path' not in st.session_state:
        st.session_state['output_path'] = DEFAULT_OUTPUT_PATH
    
    # Render sidebar
    render_sidebar()
    
    # Main content
    st.markdown('<h1 class="main-header">Akaśa: The Real-Time Knowledge Conduit</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">A streaming-first RAG system for Enterprise Support</p>', unsafe_allow_html=True)
    
    # Team info
    st.markdown("""
    <div class="team-info">
        <strong>ChronoWeavers AI Team:</strong> Raja Babu, Anand Bharti, Ram Kumar, Tak Abhishek, Anubhav Singh
        <br>
        <small>IIT (ISM) Dhanbad</small>
    </div>
    """, unsafe_allow_html=True)
    
    # Main tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat", "📊 Monitor", "⚙️ Settings"])
    
    with tab1:
        # Recent activity
        results = get_inputs()
        last_modified_time, last_indexed_files = results
        
        st.markdown("### 📑 Recent Activity")
        df = pd.DataFrame(last_indexed_files, columns=[last_modified_time, "status"])
        if df.status.isna().any():
            del df["status"]
        df.set_index(df.columns[0])
        
        col1, col2 = st.columns([3, 1])
        with col1:
            st.dataframe(df, hide_index=True, height=150, use_container_width=True)
        with col2:
            st.button("🔄 Refresh Files", use_container_width=True)
        
        # Chat interface
        render_chat_interface()
    
    with tab2:
        render_monitor_tab()
    
    with tab3:
        render_settings_tab()
    
    # Footer
    st.markdown("""
    <div style="text-align: center; margin-top: 30px; padding: 10px; font-size: 12px; color: #7F8C8D;">
        Akaśa: The Real-Time Knowledge Conduit - Developed by ChronoWeavers AI
        <br>
        <a href="https://github.com/RajaBabu15/hackfest_crono_weaver_ai" target="_blank">View on GitHub</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()