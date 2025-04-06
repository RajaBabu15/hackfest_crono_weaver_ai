import google.generativeai as genai
from dotenv import load_dotenv
import os
import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreClient

# Load environment variables
load_dotenv()

# Configure Gemini
GOOGLE_API_KEY = os.getenv('GOOGLE_API_KEY')
if not GOOGLE_API_KEY:
    raise ValueError("GOOGLE_API_KEY not found in .env file")

genai.configure(api_key=GOOGLE_API_KEY)

# First, let's check available models
try:
    # List available models
    for m in genai.list_models():
        print(f"Available model: {m.name}")
except Exception as e:
    print(f"Error listing models: {e}")

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

vector_client = VectorStoreClient(
    PATHWAY_HOST,
    PATHWAY_PORT,
    additional_headers=get_additional_headers(),
)

# Get user query
user_query = input("What do you want to know?\n\n")

# Query Pathway vector store
try:
    results = vector_client.query(
        query_text=user_query,
        n_results=1
    )
    
    # Extract documents from results
    documents = []
    for result in results:
        documents.append(result.text)
    
except Exception as e:
    print(f"Error querying Pathway: {e}")
    documents = ["No results found due to an error in querying."]

# Create prompt
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

Please proceed with your analysis and response.
"""

try:
    # Initialize Gemini model with the correct model name
    model = genai.GenerativeModel('gemini-1.5-pro')  # Use the model name from list_models()
    
    # Generate response
    response = model.generate_content(system_prompt)
    
    print("\n\n---------------------\n\n")
    print(response.text)

except Exception as e:
    print(f"An error occurred: {str(e)}")