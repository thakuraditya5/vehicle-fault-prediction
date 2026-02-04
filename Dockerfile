# Use official Python runtime as base image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    software-properties-common \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip3 install --no-cache-dir -r requirements.txt

# Copy application files
COPY api.py schemas.py app_ui.py .
COPY bcs_fault_model_combined.pkl .
COPY bcs_fault_model_MH04LQ9368.pkl .
COPY bcs_fault_model_dheradun.pkl .

# Expose ports for FastAPI (8000) and Streamlit (8501)
EXPOSE 8000 8501

# Copy a startup script
RUN echo '#!/bin/bash\nuvicorn api:app --host 0.0.0.0 --port 8000 &\nstreamlit run app_ui.py --server.port 8501 --server.address 0.0.0.0' > start.sh
RUN chmod +x start.sh

# Start both services
CMD ["./start.sh"]
