# Use a Python base image
FROM python:3.11-slim

# Set working directory in the container
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    ffmpeg \
    libsm6 \
    libxext6 \
    wget \
    && apt-get clean && rm -rf /var/lib/apt/lists/*

# Copy requirements.txt
COPY requirements.txt .

# Install Python dependencies in order of complexity
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir "numpy<2" && \
    pip install --no-cache-dir torch==2.0.0+cpu torchaudio==2.0.1+cpu --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir -r requirements.txt

# Create necessary directories
RUN mkdir -p /app/Data/Cache /app/Data/Output

# Copy the application code
COPY . .

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV NAME=AutoTranslator

# Create volume mount points
VOLUME ["/app/Data/Cache", "/app/Data/Output"]

# Run the application
CMD ["python", "main.py"]
