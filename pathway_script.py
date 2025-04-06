# pathway_script.py
import pathway as pw
import os
from sentence_transformers import SentenceTransformer
from typing import List

# --- Configuration ---
INPUT_DATA_DIR = "/app/data/input"
OUTPUT_DIR = "/app/data/output"
MODEL_NAME = 'all-MiniLM-L6-v2'

os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Pathway Schema Definition ---
class TicketSchema(pw.Schema):
    ticket_id: str
    timestamp: str
    customer_id: str
    subject: str
    body: str

# --- Embedding Function ---
print(f"Loading embedding model: {MODEL_NAME}...")
embedding_model = SentenceTransformer(MODEL_NAME)
print("Model loaded.")

class Embedder:
    def __init__(self, model):
        self.model = model

    def __call__(self, subject: str, body: str) -> List[float]:
        subject = subject or ''
        body = body or ''
        full_text = subject + " \n " + body
        embedding = self.model.encode([full_text], show_progress_bar=False).tolist()[0]
        return embedding

compute_embedding = pw.udf(Embedder(embedding_model))

# --- Pathway Pipeline Definition ---
print(f"Setting up Pathway pipeline to monitor: {INPUT_DATA_DIR}")

tickets_raw = pw.io.csv.read(
    INPUT_DATA_DIR,
    schema=TicketSchema,
    mode="streaming",
    csv_settings=pw.io.csv.CsvParserSettings(delimiter=',')
)

tickets_with_embeddings = tickets_raw.with_columns(
    embedding=compute_embedding(pw.this.subject, pw.this.body)
)

print("Pipeline configured. Will print first processed record:")
# Changed from pw.io.printout to pw.debug.printout
# pw.debug.printout(tickets_with_embeddings, limit=1, format_pretty=True)

print(f"Configuring CSV output to: {OUTPUT_DIR}")
pw.io.csv.write(
    tickets_with_embeddings,
    os.path.join(OUTPUT_DIR, "indexed_tickets.csv")
)

print("Starting Pathway pipeline processing loop...")
pw.run()
print("Pathway pipeline finished.")