# python src/ui.py

import streamlit as st
import os
from datetime import datetime
import logging
import pandas as pd
import time

from src.rag import get_chat_engine
from src.config import INPUT_DATA_DIR, PATHWAY_VECTOR_HOST, PATHWAY_VECTOR_PORT

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

st.set_page_config(page_title="Realtime RAG Assistant", layout="wide")

st.title("Realtime RAG Assistant")

@st.cache_resource
def cached_get_chat_engine():
    try:
        return get_chat_engine()
    except Exception as e:
        st.error(f"Fatal Error initializing RAG engine: {e}")
        return None

chat_engine = cached_get_chat_engine()

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message and message["sources"]:
             with st.expander("Sources"):
                 for source in message["sources"]:
                      st.caption(f"ID: {source.get('id', 'N/A')}, Score: {source.get('score', 'N/A'):.4f}")
                      st.json(source.get('metadata', {}))


if prompt := st.chat_input("Ask a question about the indexed data..."):
    if not chat_engine:
        st.error("Chat engine is not available. Please check logs.")
    else:
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        with st.chat_message("assistant"):
            message_placeholder = st.empty()
            full_response_content = ""
            sources_data = []
            try:
                with st.spinner("Thinking..."):
                    logger.info(f"Sending query to chat engine: {prompt}")
                    response = chat_engine.chat(prompt)
                    full_response_content = str(response.response)

                    if hasattr(response, 'source_nodes'):
                        sources_data = [
                            {
                                "id": node.node_id,
                                "metadata": node.metadata or {},
                                "score": node.score
                            } for node in response.source_nodes
                        ]

                    message_placeholder.markdown(full_response_content)
                    if sources_data:
                        with st.expander("Sources"):
                            for source in sources_data:
                                st.caption(f"ID: {source.get('id', 'N/A')}, Score: {source.get('score', 'N/A'):.4f}")
                                st.json(source.get('metadata', {}))

            except Exception as e:
                 full_response_content = f"Error processing query: {e}"
                 message_placeholder.error(full_response_content)
                 logger.error(f"Error during chat: {e}", exc_info=True)

        st.session_state.messages.append({
            "role": "assistant",
            "content": full_response_content,
            "sources": sources_data
        })


with st.sidebar:
    st.header("System Status")
    st.write(f"Input Directory: `{INPUT_DATA_DIR}`")
    st.write(f"Vector Store: `{PATHWAY_VECTOR_HOST}:{PATHWAY_VECTOR_PORT}`")

    st.subheader("Input Monitor")
    placeholder = st.empty()
    placeholder_output = st.empty()

    if st.button("Refresh Status"):
        try:
            files = [f for f in os.listdir(INPUT_DATA_DIR) if os.path.isfile(os.path.join(INPUT_DATA_DIR, f))]
            files.sort(key=lambda x: os.path.getmtime(os.path.join(INPUT_DATA_DIR, x)), reverse=True)

            recent_files_info = []
            for f in files[:5]:
                 f_path = os.path.join(INPUT_DATA_DIR, f)
                 mod_time = datetime.fromtimestamp(os.path.getmtime(f_path)).strftime('%Y-%m-%d %H:%M:%S')
                 recent_files_info.append({"file": f, "modified": mod_time})

            df = pd.DataFrame(recent_files_info)
            placeholder.dataframe(df, hide_index=True, use_container_width=True)

        except Exception as e:
            placeholder.error(f"Failed to scan input dir: {e}")

        # Basic check if Vector Store is reachable (requires Pathway client library interaction or separate health check)
        # This is a placeholder - real check needs Pathway client interaction
        try:
            # Attempt a basic connection or status check if PathwayVectorStore provides one
            # For now, just assume it's running if the app starts
            if chat_engine: # Check if engine initialized which implies store was reachable at start
                 st.success("Vector Store assumed reachable.")
            else:
                 st.warning("Vector Store may not be reachable (engine init failed).")

        except Exception as e:
             st.error(f"Vector store check failed: {e}")