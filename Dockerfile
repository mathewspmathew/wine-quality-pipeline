# Use Python base image
FROM python:3.11-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    nginx \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /opt/ml

# Copy requirements
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY src/ ./code/src/
COPY models/ ./code/models/

# Train model
WORKDIR /opt/ml/code
RUN python -m src.train || true

# Create serve script for SageMaker
RUN echo '#!/bin/bash\n\
cd /opt/ml/code\n\
uvicorn src.app:app --host 0.0.0.0 --port 8080' > /opt/ml/code/serve && \
    chmod +x /opt/ml/code/serve

# Expose port
EXPOSE 8080

# Set environment variables
ENV PYTHONUNBUFFERED=TRUE

# SageMaker uses /opt/ml/code/serve to start the service
ENTRYPOINT ["/opt/ml/code/serve"]