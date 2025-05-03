# Use a lightweight base image with Python
FROM python:3.10-slim

# Avoids interactive prompts during package install
ENV DEBIAN_FRONTEND=noninteractive

# Install system dependencies
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    poppler-utils \
    && rm -rf /var/lib/apt/lists/*

# Optional: set environment variable to suppress HF tokenizer warnings
ENV TOKENIZERS_PARALLELISM=false

# Set working directory
WORKDIR /app

# Copy and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . .

# Expose Flask port
EXPOSE 8000

# Default command (can be overridden)
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:8000", "src.app:app"]
