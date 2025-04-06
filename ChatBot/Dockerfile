FROM python:3.10-slim

WORKDIR /app

# Install required packages
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your code and data
COPY *.py .
COPY .env .
RUN mkdir -p Data

# Expose the Streamlit port
EXPOSE 8501
# Expose the Pathway port
EXPOSE 8000

# Set environment variables for Streamlit
ENV STREAMLIT_SERVER_PORT=8501
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ENABLE_CORS=false

# Run the Streamlit application
CMD ["streamlit", "run", "app.py"]