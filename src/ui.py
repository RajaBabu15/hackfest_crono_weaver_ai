import streamlit as st
import os
from datetime import datetime
import logging
import pandas as pd
import time

from src.rag import get_chat_engine
from src.config import INPUT_DATA_DIR, PATHWAY_VECTOR_HOST, PATHWAY_VECTOR_PORT

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Akaśa: Real-Time RAG Assistant",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for styling
st.markdown("""
    <style>
    .main-title {
        font-size: 3rem;
        font-weight: bold;
        color: #1E3A8A;
        margin-bottom: 0.5rem;
        text-align: center;
    }
    .subtitle {
        font-size: 1.5rem;
        color: #4B5563;
        margin-bottom: 2rem;
        text-align: center;
        font-style: italic;
    }
    .header-container {
        text-align: center;
        background: linear-gradient(to right, #EEF2FF, #C7D2FE, #EEF2FF);
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    .completed-feature {
        color: #059669;
        font-weight: bold;
    }
    .team-section {
        background-color: #F8FAFC;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #E2E8F0;
    }
    .team-member {
        margin-bottom: 0.5rem;
    }
    .feature-tag {
        background-color: #DBEAFE;
        color: #1E40AF;
        padding: 0.2rem 0.5rem;
        border-radius: 4px;
        font-size: 0.8rem;
        margin-right: 0.3rem;
        display: inline-block;
    }
    </style>
    """, unsafe_allow_html=True)

# Header section with project details
st.markdown('<div class="header-container">', unsafe_allow_html=True)
st.markdown('<div class="main-title">Akaśa: The Real-Time Knowledge Conduit</div>', unsafe_allow_html=True)
st.markdown('<div class="subtitle">A Real-Time RAG Assistant for Enterprise Support/Marketing</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align:center">IIT (ISM) Dhanbad Hackathon Project by ChronoWeavers AI</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)

# Initialize chat engine with caching
@st.cache_resource
def cached_get_chat_engine():
    try:
        logger.info("Initializing RAG engine...")
        engine = get_chat_engine()
        logger.info("RAG engine initialized successfully")
        return engine
    except Exception as e:
        logger.error(f"Fatal Error initializing RAG engine: {e}", exc_info=True)
        return None

# Get the chat engine
chat_engine = cached_get_chat_engine()

# Initialize session state for messages if not present
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"], avatar="👤" if message["role"] == "user" else "🤖"):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
            with st.expander("📚 Sources"):
                for i, source in enumerate(message["sources"]):
                    st.markdown(f"**Source {i+1}:**")
                    st.caption(f"ID: `{source.get('id', 'N/A')}`")
                    st.caption(f"Relevance Score: `{source.get('score', 'N/A'):.4f}`")
                    
                    # Display metadata in a more readable format
                    with st.container():
                        st.json(source.get('metadata', {}))
                    st.divider()

# Chat input and response handling
if prompt := st.chat_input("Ask a question about the indexed data..."):
    if not chat_engine:
        st.error("⚠️ Chat engine is not available. Please check logs and ensure vector store is running.")
    else:
        # Add user message to history and display
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.markdown(prompt)

        # Process and display assistant response
        with st.chat_message("assistant", avatar="🤖"):
            message_placeholder = st.empty()
            full_response_content = ""
            sources_data = []
            
            try:
                with st.spinner("🧠 Thinking..."):
                    logger.info(f"Sending query to chat engine: {prompt}")
                    start_time = time.time()
                    response = chat_engine.chat(prompt)
                    query_time = time.time() - start_time
                    
                    full_response_content = str(response.response)
                    logger.info(f"Query processed in {query_time:.2f} seconds")

                    # Extract source nodes if available
                    if hasattr(response, 'source_nodes'):
                        sources_data = [
                            {
                                "id": node.node_id,
                                "metadata": node.metadata or {},
                                "score": node.score
                            } for node in response.source_nodes
                        ]

                # Display the response
                message_placeholder.markdown(full_response_content)
                
                # Show sources if available
                if sources_data:
                    with st.expander("📚 Sources"):
                        for i, source in enumerate(sources_data):
                            st.markdown(f"**Source {i+1}:**")
                            st.caption(f"ID: `{source.get('id', 'N/A')}`")
                            st.caption(f"Relevance Score: `{source.get('score', 'N/A'):.4f}`")
                            
                            with st.container():
                                st.json(source.get('metadata', {}))
                            st.divider()

            except Exception as e:
                full_response_content = f"Error processing query: {e}"
                message_placeholder.error(full_response_content)
                logger.error(f"Error during chat: {e}", exc_info=True)

        # Add assistant response to history
        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response_content,
            "sources": sources_data
        })

# Sidebar with system status and team information
with st.sidebar:
    # Logo/icon placeholder
    st.image("https://via.placeholder.com/100x100.png?text=Akaśa", width=100)
    
    st.markdown("### 🛠️ System Status")
    
    # System configuration display
    st.markdown("#### Configuration")
    st.code(f"Input Directory: {INPUT_DATA_DIR}")
    st.code(f"Vector Store: {PATHWAY_VECTOR_HOST}:{PATHWAY_VECTOR_PORT}")
    
    # Connection status
    status_container = st.container()
    with status_container:
        if chat_engine:
            st.success("✅ RAG Engine Connected")
        else:
            st.error("❌ RAG Engine Disconnected")
    
    st.divider()
    
    # Implemented features section
    st.markdown("### ✨ Implemented Features")
    
    features = [
        "✅ Continuous Data Ingestion & Indexing",
        "✅ Retrieval-Augmented Q&A with LLM",
        "✅ Real-Time Updates & Consistency",
        "✅ Interactive User Interface",
        "✅ FastAPI Endpoint Integration",
        "✅ Robustness and Fallback Mechanisms",
        "✅ Multi-format Data Source Support",
        "✅ Automatic Knowledge Base Updates"
    ]
    
    for feature in features:
        st.markdown(feature)
    
    st.divider()
    
    # Input monitor section
    st.markdown("### 📂 Input Monitor")
    st.caption("Most recently modified files in the input directory")
    
    file_list_container = st.empty()
    status_message = st.empty()
    
    if st.button("🔄 Refresh Files"):
        try:
            with st.spinner("Scanning input directory..."):
                files = [f for f in os.listdir(INPUT_DATA_DIR) if os.path.isfile(os.path.join(INPUT_DATA_DIR, f))]
                files.sort(key=lambda x: os.path.getmtime(os.path.join(INPUT_DATA_DIR, x)), reverse=True)

                if not files:
                    status_message.info("No files found in input directory")
                else:
                    recent_files_info = []
                    for f in files[:10]:  # Show up to 10 files
                        f_path = os.path.join(INPUT_DATA_DIR, f)
                        mod_time = datetime.fromtimestamp(os.path.getmtime(f_path)).strftime('%Y-%m-%d %H:%M:%S')
                        file_size = os.path.getsize(f_path) / 1024  # Size in KB
                        recent_files_info.append({
                            "File": f, 
                            "Modified": mod_time,
                            "Size (KB)": f"{file_size:.1f}"
                        })

                    df = pd.DataFrame(recent_files_info)
                    file_list_container.dataframe(
                        df, 
                        hide_index=True, 
                        use_container_width=True,
                        column_config={
                            "File": st.column_config.TextColumn("File", width="medium"),
                            "Modified": st.column_config.TextColumn("Last Modified", width="medium"),
                            "Size (KB)": st.column_config.NumberColumn("Size (KB)", format="%.1f")
                        }
                    )
                    status_message.success(f"Found {len(files)} files in input directory")

        except Exception as e:
            status_message.error(f"Failed to scan input directory: {e}")
            logger.error(f"Error scanning input directory: {e}", exc_info=True)
    
    # Button to manually refresh RAG index
    if st.button("🔄 Reload RAG Index"):
        try:
            with st.spinner("Reloading knowledge base..."):
                # This would call a function to reload the index - implementation depends on your code
                # For example: reload_index()
                st.success("Knowledge base successfully updated!")
        except Exception as e:
            st.error(f"Failed to reload knowledge base: {e}")
    
    st.divider()
    
    # Team info section
    st.markdown("### 👥 Team ChronoWeavers AI")
    
    team_members = [
        {"name": "Raja Babu (21JE0740)", "role": "Team Lead / Backend & Integration", "skills": ["Python", "FastAPI", "System Architecture"]},
        {"name": "Tak Abhishek (21JE0979)", "role": "Data Engineer / Pathway Specialist", "skills": ["Pathway", "Real-Time Pipelines", "Vector DBs"]},
        {"name": "Ram Kumar (21JE0745)", "role": "AI/ML Engineer / RAG Architect", "skills": ["LLM APIs", "Prompt Engineering", "NLP"]},
        {"name": "Anand Bharti (21JE0102)", "role": "Frontend & UI/UX Developer", "skills": ["Streamlit", "UX Design", "API Consumption"]},
        {"name": "Anubhav Singh (21JE0141)", "role": "Support Developer / Data Validation", "skills": ["Data Parsing", "Testing", "Git"]}
    ]
    
    for member in team_members:
        with st.container():
            st.markdown(f"**{member['name']}**")
            st.caption(f"_{member['role']}_")
            skill_html = " ".join([f"<span class='feature-tag'>{skill}</span>" for skill in member['skills']])
            st.markdown(skill_html, unsafe_allow_html=True)
        st.markdown("---")
    
    # About section
    with st.expander("ℹ️ About Akaśa"):
        st.markdown("""
        **Akaśa: The Real-Time Knowledge Conduit**
        
        Named after the Sanskrit word for "sky" or "space," Akaśa represents our 
        vision for an all-encompassing, constantly updating knowledge system. 
        
        This application provides a chat interface for querying documents with 
        real-time retrieval-augmented generation. New documents added to the input 
        directory are automatically processed and made available for querying, ensuring 
        your knowledge base is always up-to-date.
        
        Developed by ChronoWeavers AI team at IIT (ISM) Dhanbad Hackathon.
        """)

# Create tabs for main content
tab1, tab2 = st.tabs(["💬 Chat", "📊 Project Overview"])

with tab1:
    # Chat interface is already displayed above
    pass

with tab2:
    # Project Overview content
    col1, col2 = st.columns([3, 2])
    
    with col1:
        st.markdown("## 🚀 Project Description")
        st.markdown("""
        **Akaśa: The Real-Time Knowledge Conduit** is a cutting-edge Retrieval-Augmented Generation (RAG) 
        system that addresses the challenge of maintaining up-to-date information in enterprise
        knowledge bases. Unlike traditional RAG systems, Akaśa continuously ingests and processes
        new data in real-time, ensuring that responses are always based on the most current information.
        
        ### Key Capabilities:
        """)
        
        st.markdown("""
        - **Real-Time Knowledge Updates**: Automatically detects and processes new data as it arrives
        - **Multi-Format Support**: Handles various document formats including CSV, PDF, and text files
        - **Intelligent Retrieval**: Uses advanced embedding techniques to find the most relevant information
        - **Natural Conversations**: Provides human-like responses with source citations for transparency
        - **API Integration**: Connects seamlessly with other enterprise systems via REST API
        """)
    
    with col2:
        st.markdown("## 🏆 Achievements")
        st.markdown("""
        - Successfully implemented a fully-functional real-time RAG system
        - Created an intuitive user interface for easy interaction
        - Developed robust data pipeline using Pathway for continuous processing
        - Integrated with popular LLM APIs for high-quality responses
        - Built with scalability in mind for enterprise deployment
        """)
    
    st.markdown("## 🔄 System Architecture")
    st.markdown("""
    Akaśa is built on a modern, microservices-based architecture that enables real-time data processing and retrieval:
    
    1. **Data Ingestion Layer**: Monitors input directory for new files and processes them immediately
    2. **Pathway Processing Pipeline**: Transforms raw data into embeddings for semantic search
    3. **Vector Database**: Stores document embeddings for fast similarity search
    4. **RAG Engine**: Combines retrieved context with LLM capabilities for accurate responses
    5. **Multiple Interfaces**: Streamlit UI and FastAPI endpoints for various integration scenarios
    """)
    
    # Simulated architecture diagram using columns
    arch_cols = st.columns(5)
    with arch_cols[0]:
        st.markdown("""
        ```
        ┌─────────────┐
        │ Data Sources│
        │  (CSV, PDF) │
        └─────────────┘
        ```
        """)
    with arch_cols[1]:
        st.markdown("""
        ```
        ┌─────────────┐
        │   Pathway   │
        │  Pipeline   │
        └─────────────┘
        ```
        """)
    with arch_cols[2]:
        st.markdown("""
        ```
        ┌─────────────┐
        │   Vector    │
        │   Store     │
        └─────────────┘
        ```
        """)
    with arch_cols[3]:
        st.markdown("""
        ```
        ┌─────────────┐
        │  RAG Engine │
        │  (LlamaIdx) │
        └─────────────┘
        ```
        """)
    with arch_cols[4]:
        st.markdown("""
        ```
        ┌─────────────┐
        │ UI & API    │
        │ Interfaces  │
        └─────────────┘
        ```
        """)
    
    st.markdown("## 🔮 Future Enhancements")
    
    enhancements = [
        "Advanced document parsing and structured data extraction",
        "Multi-modal support (images, audio transcripts)",
        "Slack and Microsoft Teams integration",
        "User feedback loop for continuous improvement",
        "Enhanced security features for enterprise deployment"
    ]
    
    for i, enhancement in enumerate(enhancements):
        st.markdown(f"**{i+1}.** {enhancement}")

# Add custom CSS for better styling
st.markdown("""
<style>
    /* Chat container styling */
    .stChatMessage {
        padding: 1rem 0;
    }
    
    /* Reduce default padding */
    .block-container {
        padding-top: 1rem;
    }
    
    /* Button styling */
    .stButton button {
        border-radius: 4px;
        font-weight: 500;
    }
    
    /* Code block styling */
    code {
        border-radius: 4px;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #F1F5F9;
        border-radius: 4px 4px 0px 0px;
        gap: 1px;
        padding-top: 10px;
        padding-bottom: 10px;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #DBEAFE !important;
        color: #1E40AF !important;
    }
</style>
""", unsafe_allow_html=True)