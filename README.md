
# Akaśa: The Real-Time Knowledge Conduit

![Akaśa Logo](assets/logo.png)

## About the Project

**Akaśa**, named after the Sanskrit word for "sky" or "space," represents our vision for an all-encompassing, constantly updating knowledge system. This real-time RAG (Retrieval-Augmented Generation) assistant addresses the challenge of maintaining up-to-date information in enterprise knowledge bases.

Unlike traditional RAG systems, Akaśa continuously ingests and processes new data in real-time, ensuring that responses are always based on the most current information.

**Project:** Real-Time RAG Assistant for Enterprise Support/Marketing  
**Team:** ChronoWeavers AI  
**Institution:** IIT (ISM) Dhanbad Hackathon

## Key Features

- ✅ **Continuous Data Ingestion & Indexing:** Automatically monitors and processes new data as it arrives.
- ✅ **Retrieval-Augmented Q&A with LLM:** Provides human-like responses with source citations.
- ✅ **Real-Time Updates & Consistency:** Updates the knowledge base without requiring an application restart.
- ✅ **Interactive User Interface:** Clean Streamlit UI for easy interaction.
- ✅ **FastAPI Endpoint Integration:** REST API for seamless integration with other systems.
- ✅ **Robustness and Fallback Mechanisms:** Includes error handling and graceful degradation.
- ✅ **Multi-format Data Source Support:** Handles various document formats.
- ✅ **Automatic Knowledge Base Updates:** Real-time index maintenance.

## System Architecture

Akaśa is built on a modern, microservices-based architecture:

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│ Data Sources│    │   Pathway   │    │   Vector    │    │  RAG Engine │    │ UI & API    │
│  (CSV, PDF) │───▶│  Pipeline   │───▶│   Store     │───▶│  (LlamaIdx) │───▶│ Interfaces  │
└─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘
```

## Installation and Setup

### Prerequisites

- Python 3.9+
- Docker (optional, for containerized deployment)
- OpenAI API key (for LLM access)

### Environment Setup

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/chronoweavers/akasa-rag.git
   cd akasa-rag
   ```

2. **Create and Activate a Virtual Environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install Required Packages:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set Up Environment Variables:**
   ```bash
   cp .env.example .env
   # Edit .env with your OpenAI API key and other settings
   ```

## Running the Application

1. **Start the Pathway Pipeline for Data Processing:**
   ```bash
   python -m src.run_pipeline
   ```

2. **Start the Streamlit UI in a Separate Terminal:**
   ```bash
   streamlit run src/ui.py
   ```

3. **(Optional) Start the FastAPI Server:**
   ```bash
   uvicorn src.api:app --host 0.0.0.0 --port 8000
   ```

## Docker Deployment

### Building the Docker Image

To build the Docker image, run:

```bash
sudo docker build -t realtime-rag-assistant .
```

**Note:** During the build, you may see a warning like:  
```
failed to fetch metadata: fork/exec /usr/local/lib/docker/cli-plugins/docker-buildx: no such file or directory
```
This warning indicates that Docker Buildx (BuildKit) is not installed on your system. Although the legacy builder is still supported, it is deprecated and will be removed in future releases. We recommend installing Buildx following the instructions at [Docker’s Buildx documentation](https://docs.docker.com/go/buildx/).

### Running the Container

After building the image, you can run the container with:

```bash
sudo docker run -d --rm \
    --name realtime-rag \
    -p 8501:8501 \
    -p 8000:8000 \
    -v "$(pwd)/data/input:/app/data/input" \
    -v "$(pwd)/data/output:/app/data/output" \
    --env-file .env \
    realtime-rag-assistant
```

Monitor the logs to ensure everything starts correctly:

```bash
sudo docker logs -f <container_id>
```

You should see an output similar to:
```
Starting Pathway Pipeline...
Pathway Pipeline PID: 7
```

## Usage Guide

### Adding Data Sources

1. Place your data files (CSV, TXT, PDF) in the configured input directory.
2. The system will automatically detect, process, and index the new data.
3. The knowledge base is updated in real-time (no restart required).

### Using the Chat Interface

1. Navigate to the Streamlit UI (default: [http://localhost:8501](http://localhost:8501)).
2. Type your query in the input box.
3. The system retrieves relevant information and generates a response.
4. Sources used to generate the response are displayed in an expandable section.

### API Usage

```python
import requests

response = requests.post(
    "http://localhost:8000/query",
    json={"query": "What are the latest customer support issues?"}
)

print(response.json())
```

## Troubleshooting

- **Docker Buildx Warning:**  
  If you see a warning about missing `docker-buildx`, refer to the [Buildx installation guide](https://docs.docker.com/go/buildx/) to set up the new build system.
  
- **Data Ingestion Issues:**  
  Ensure that your data files are correctly formatted and placed in the designated input directory. The system logs will provide information if any errors occur during ingestion.

## Team Members

| Name                | Role                                | Key Skills                                                   |
|---------------------|-------------------------------------|--------------------------------------------------------------|
| Raja Babu (21JE0740)| Team Lead / Backend & Integration   | Python, FastAPI, API Design, Slack SDK, System Architecture    |
| Tak Abhishek (21JE0979)| Data Engineer / Pathway Specialist | Pathway, Real-Time Pipelines, Kafka/Streaming, Vector DBs, Python|
| Ram Kumar (21JE0745) | AI/ML Engineer / RAG Architect      | Python, RAG Strategy, LLM APIs (OpenAI/Anthropic), Prompt Engineering, NLP |
| Anand Bharti (21JE0102)| Frontend & UI/UX Developer          | HTML/CSS/JS, Streamlit/Gradio, API Consumption, UX Design         |
| Anubhav Singh (21JE0141)| Support Developer / Data Validation | Python, Data Parsing (PDFs, JSON), Testing, Git                    |

## Future Enhancements

- Advanced document parsing and structured data extraction.
- Multi-modal support (images, audio transcripts).
- Slack and Microsoft Teams integration.
- User feedback loop for continuous improvement.
- Enhanced security features for enterprise deployment.

## License

MIT License. See [LICENSE](LICENSE) for more information.

## Acknowledgements

- [Pathway](https://pathway.com/) for real-time data processing capabilities.
- [LlamaIndex](https://www.llamaindex.ai/) for RAG framework.
- [Streamlit](https://streamlit.io/) for the UI framework.

