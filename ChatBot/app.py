import streamlit as st
import google.generativeai as genai
from dotenv import load_dotenv
import os
import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreClient

# Load environment variables
load_dotenv()

# Page configuration
st.set_page_config(
    page_title="Document AI Assistant",
    page_icon="📚",
    layout="centered",
)

# Styling
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .stTextInput>div>div>input {
        border-radius: 10px;
    }
    .stButton>button {
        border-radius: 10px;
        background-color: #4CAF50;
        color: white;
        border: none;
    }
</style>
""", unsafe_allow_html=True)

# Header
st.title("📚 Document AI Assistant")
st.markdown("Ask questions about your documents and get AI-powered responses.")

# Initialize session state for chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Configure Gemini
@st.cache_resource
def initialize_genai():
    GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
    if not GOOGLE_API_KEY:
        st.error("GOOGLE_API_KEY not found in .env file")
        return None
    
    genai.configure(api_key=GOOGLE_API_KEY)
    return genai.GenerativeModel('gemini-1.5-pro')

# Initialize Pathway client
@st.cache_resource
def initialize_pathway():
    # Setting the environment for Pathway
    PATHWAY_HOST = os.environ.get("PATHWAY_HOST", "localhost")
    PATHWAY_PORT = int(os.environ.get("PATHWAY_PORT", "8000"))
    PATHWAY_API_KEY = os.environ.get("PATHWAY_API_KEY")

    # Initialize Pathway VectorStoreClient
    def get_additional_headers():
        headers = {}
        if PATHWAY_API_KEY is not None:
            headers = {"X-Pathway-API-Key": PATHWAY_API_KEY}
        return headers

    return VectorStoreClient(
        PATHWAY_HOST,
        PATHWAY_PORT,
        additional_headers=get_additional_headers(),
    )

# Initialize the resources
genai_model = initialize_genai()
vector_client = initialize_pathway()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Query function
def query_documents(user_query):
    try:
        results = vector_client.query(
            query_text=user_query,
            n_results=3  # Get top 3 relevant chunks
        )
        
        # Extract documents from results
        documents = []
        for result in results:
            documents.append(result.text)
        
        return documents
    except Exception as e:
        st.error(f"Error querying Pathway: {e}")
        return ["No results found due to an error in querying."]

# Process query and generate response
def generate_response(user_query, documents):
    system_prompt = f"""
    You are an expert assisstant on various domains. Your role is to provide precise, data-driven answers based solely on the provided context.

    REFERENCE MATERIAL:
    -------------------
    {str(documents)}

    QUERY FOR ANALYSIS:
    -------------------
    {user_query}

    REQUIRED ANALYSIS FRAMEWORK:
    ---------------------------
    1. CONTEXT SCANNING
    * Identify all passages relevant to the query
    * Extract specific data points, measurements, and recommendations
    * Note any contextual conditions or limitations

    2. DATA RELEVANCE ASSESSMENT
    * Rate the relevance of found information (High/Medium/Low)
    * Identify any information gaps
    * Flag any ambiguous or incomplete information

    3. SYNTHESIS AND FORMULATION
    * Compile relevant information into a coherent answer
    * Structure the response logically
    * Include specific examples and measurements where available
    * Note any important qualifications or conditions

    RESPONSE STRUCTURE:
    ------------------
    1. Direct Answer:
    [Provide the most direct answer based on available information]

    2. Supporting Details:
    * [List specific details from the context]
    * [Include relevant measurements/timing/conditions]

    3. Important Considerations:
    * [List any caveats or special conditions]
    * [Note any contextual limitations]

    4. Information Gaps:
    * [Note any aspects of the question not addressed in the context]

    MANDATORY RULES:
    ---------------
    1. Only use information from the provided context
    2. Do not incorporate external knowledge
    3. Clearly indicate when information is incomplete or missing
    4. Use precise quotes where appropriate
    5. Maintain scientific accuracy

    If insufficient information exists in the context, state: "The provided context does not contain sufficient information to answer this question about [specific topic]."

    Please provide your answer in markdown format for better readability.
    """

    try:
        response = genai_model.generate_content(system_prompt)
        return response.text
    except Exception as e:
        return f"An error occurred while generating the response: {str(e)}"

# User input
user_query = st.chat_input("Ask a question about your documents")

if user_query:
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": user_query})
    
    # Display user message
    with st.chat_message("user"):
        st.markdown(user_query)
    
    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Searching documents and generating response..."):
            # Query documents
            documents = query_documents(user_query)
            
            # Generate response
            response = generate_response(user_query, documents)
            
            # Display response
            st.markdown(response)
    
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})

# Sidebar
with st.sidebar:
    st.title("About")
    st.markdown("This AI assistant uses Pathway for document retrieval and Google's Gemini for generating responses.")
    
    st.markdown("### Document Statistics")
    try:
        stats = "Retrieving statistics..."
        # You could add actual statistics from Pathway here if available
        st.markdown(stats)
    except:
        st.markdown("Unable to retrieve document statistics")
    
    # Reset conversation button
    if st.button("Reset Conversation"):
        st.session_state.messages = []
        st.rerun()