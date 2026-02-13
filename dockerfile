FROM python:3.11-slim

WORKDIR /app

# Install system dependencies required for OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY . .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Ensure results folders exist
RUN mkdir -p results/defects

# Use ENTRYPOINT so CLI args work properly
ENTRYPOINT ["python", "src/main.py"]
