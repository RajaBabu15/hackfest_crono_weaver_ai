# Use the official Pathway image as the base
FROM pathwaycom/pathway:latest

# Set the working directory inside the container
WORKDIR /app

# Install specific dependencies needed by the scripts
# This includes Streamlit for the web interface
RUN pip install --no-cache-dir sentence-transformers torch streamlit

# Copy the Python scripts into the working directory
COPY pathway_script.py .
COPY streamlit_app.py .

# Pre-download the Sentence Transformer model during the build phase
# This avoids downloading it every time the container runs and makes startup faster
RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# Create the expected data directories within the container image
RUN mkdir -p /app/data/input /app/data/output

# Expose the port that Streamlit will run on
EXPOSE 8501

# Create a startup script to run both Pathway and Streamlit
RUN echo '#!/bin/bash\npython ./pathway_script.py & streamlit run streamlit_app.py' > /app/start.sh && \
    chmod +x /app/start.sh

# Command to run when the container starts
CMD ["/app/start.sh"]











# # Use the official Pathway image as the base
# FROM pathwaycom/pathway:latest

# # Set the working directory inside the container
# WORKDIR /app

# # Install specific dependencies needed by the script
# # sentence-transformers requires torch (PyTorch)
# # Using --no-cache-dir saves space in the final image layer.
# RUN pip install --no-cache-dir sentence-transformers torch

# # Copy the Python script into the working directory
# COPY pathway_script.py .

# # Pre-download the Sentence Transformer model during the build phase
# # This avoids downloading it every time the container runs and makes startup faster.
# # Replace 'all-MiniLM-L6-v2' if you changed the model name.
# RUN python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('all-MiniLM-L6-v2')"

# # Create the expected data directories within the container image
# # Although volumes will overlay these, creating them makes intent clearer
# # and ensures they exist if volumes aren't mounted for some reason.
# RUN mkdir -p /app/data/input /app/data/output

# # Command to run when the container starts
# CMD ["python", "./pathway_script.py"]