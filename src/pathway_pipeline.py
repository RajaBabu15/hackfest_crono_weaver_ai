# python src/pathway_pipeline.py

import pathway as pw
from sentence_transformers import SentenceTransformer
import pandas as pd
import os
from typing import List # Import List if not already there

from src import config

OUTPUT_CSV_PATH = "/app/data/output/indexed_data.csv"

class TicketSchema(pw.Schema):
    ticket_id: str
    timestamp: str
    customer_id: str
    subject: str
    body: str

print(f"Loading embedding model: {config.EMBEDDING_MODEL_NAME}...")
embedding_model = SentenceTransformer(config.EMBEDDING_MODEL_NAME)
print("Model loaded.")

# Modify the UDF to operate on column inputs, not DataFrames
class EmbedderForRow:
    def __init__(self, model):
        self.model = model

    # __call__ now takes individual column values from a row
    def __call__(self, subject: str, body: str) -> str:
        # Process a single row's data
        subject = subject or ''
        body = body or ''
        full_text = subject + " \n " + body
        # Encode expects a list, even if it's just one item
        embedding_vector = self.model.encode([full_text], show_progress_bar=False)[0]
        # Return the string representation of the list for this single row
        return str(embedding_vector.tolist())

# Wrap the row-based embedder
compute_embedding_for_row = pw.udf(EmbedderForRow(embedding_model))

print(f"Setting up Pathway pipeline to monitor: {config.INPUT_DATA_DIR}")

tickets_raw = pw.io.fs.read(
    config.INPUT_DATA_DIR,
    schema=TicketSchema,
    format="csv",
    mode="streaming",
    with_metadata=True,
    csv_settings=pw.io.csv.CsvParserSettings(delimiter=',')
)

# Use with_columns to apply the row-based UDF
enriched_tickets = tickets_raw.with_columns(
    # Pass the relevant columns for the current row (pw.this) to the UDF
    embedding_str=compute_embedding_for_row(pw.this.subject, pw.this.body)
)

print(f"Configuring CSV writer to: {OUTPUT_CSV_PATH}")

# --- Explicitly select ONLY the columns needed for the CSV output ---
output_table = enriched_tickets.select(
    pw.this.ticket_id,
    pw.this.timestamp,
    pw.this.customer_id,
    pw.this.subject,
    pw.this.body,
    pw.this.embedding_str # The embedding string column
)
# -------------------------------------------------------------------

print(f"Configuring CSV writer to: {OUTPUT_CSV_PATH}")

# Write the selected table, not the one with potential extra columns
pw.io.csv.write(
    output_table,
    OUTPUT_CSV_PATH
)

print("Starting Pathway pipeline processing loop...")
pw.run()
print("Pathway pipeline finished.")

'''
sudo docker build -t realtime-rag-assistant .

sudo docker run -d --rm \
    --name realtime-rag \
    -p 8501:8501 \
    -p 8000:8000 \
    -v "$(pwd)/data/input:/app/data/input" \
    -v "$(pwd)/data/output:/app/data/output" \
    --env-file .env \
    realtime-rag-assistant

    
# View logs (follow)
sudo docker logs -f realtime-rag

# Stop the container (when running detached)
sudo docker stop realtime-rag


"What tickets mention payment?"
"Find tickets with high urgency."
"Tell me about TKT-000000" (or another ID from your simulation).
'''