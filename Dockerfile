FROM pathwaycom/pathway:latest

WORKDIR /app

ARG NB_USER=pathway
ARG NB_UID=1000
ARG NB_GID=1000
ARG EMBEDDING_MODEL_NAME=all-MiniLM-L6-v2

RUN groupadd -g ${NB_GID} ${NB_USER} && \
    useradd -u ${NB_UID} -g ${NB_GID} -m ${NB_USER}

RUN apt-get update && \
    apt-get install -y --no-install-recommends build-essential && \
    rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --upgrade pip --no-cache-dir && \
    pip install --no-cache-dir -r requirements.txt && \
    python --version

COPY ./src ./src
COPY ./scripts ./scripts
COPY ./start.sh .

RUN mkdir -p /app/data/input /app/data/output && \
    chown -R ${NB_UID}:${NB_GID} /app/data

RUN chmod +x /app/start.sh

USER ${NB_USER}

# ----> Add Step to clear cache <----
RUN echo "Clearing Hugging Face cache..." && \
    rm -rf /home/${NB_USER}/.cache/huggingface && \
    echo "Cache cleared."
# ------------------------------------

# Pre-download the embedding model AS THE RUNTIME USER
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('${EMBEDDING_MODEL_NAME}')"

EXPOSE 8501
EXPOSE 8000

CMD ["/app/start.sh"]