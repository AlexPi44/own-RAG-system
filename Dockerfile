# Dockerfile for Super Creator Agent on Hugging Face Spaces
FROM python:3.10-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    git \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Install Ollama
RUN curl -fsSL https://ollama.com/install.sh | sh

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY super_creator_agent.py .
COPY app.py .

# Create necessary directories
RUN mkdir -p /app/sca_db /app/uploaded_docs /app/sca_project /app/flashrank_cache

# Expose Gradio port
EXPOSE 7860

# Start Ollama in background and pull models, then run app
CMD ollama serve & \
    sleep 10 && \
    ollama pull qwen2.5-coder:3b && \
    ollama pull qwen2.5-coder:7b && \
    python app.py