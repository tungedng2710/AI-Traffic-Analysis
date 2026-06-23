FROM paddlepaddle/paddle:3.3.1-gpu-cuda12.9-cudnn9.9

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libgl1 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# Set work directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Expose ports for both services
EXPOSE 7862 7863

# Default command (can be overridden in docker-compose)
CMD ["python", "main.py"]
