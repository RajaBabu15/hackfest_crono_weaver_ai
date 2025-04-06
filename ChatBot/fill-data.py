from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
import pathway as pw
from pathway.xpacks.llm.vector_store import VectorStoreClient
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Setting the environment
DATA_PATH = r"Data"

# Define Pathway connection parameters
PATHWAY_HOST = os.environ.get("PATHWAY_HOST", "localhost")
PATHWAY_PORT = int(os.environ.get("PATHWAY_PORT", "8000"))
PATHWAY_API_KEY = os.environ.get("PATHWAY_API_KEY")

# Create Vector Store Client
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

# Load documents
loader = PyPDFDirectoryLoader(DATA_PATH)
raw_documents = loader.load()
print(f"Loaded {len(raw_documents)} documents")

# Split the documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=100,
    length_function=len,
    is_separator_regex=False,
)

chunks = text_splitter.split_documents(raw_documents)
print(f"Created {len(chunks)} chunks")

# Prepare documents for indexing
documents = []
for i, chunk in enumerate(chunks):
    documents.append({
        "id": f"ID{i}",
        "text": chunk.page_content,
        "metadata": chunk.metadata
    })

# Upload documents to Pathway
try:
    for doc in documents:
        vector_client.index_document(
            doc_id=doc["id"],
            content=doc["text"],
            metadata=doc["metadata"]
        )
    print(f"Successfully indexed {len(documents)} documents to Pathway")
except Exception as e:
    print(f"Error indexing documents to Pathway: {e}")