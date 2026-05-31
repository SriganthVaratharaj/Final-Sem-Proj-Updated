# Use a lightweight python image
FROM python:3.12-slim

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1

# Install system dependencies (libgl1 is required for OpenCV)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend files into the backend/ folder
COPY . backend/

# Expose Hugging Face default port
EXPOSE 7860

# Run uvicorn pointing to backend.main:app
CMD ["uvicorn", "backend.main:app", "--host", "0.0.0.0", "--port", "7860"]
