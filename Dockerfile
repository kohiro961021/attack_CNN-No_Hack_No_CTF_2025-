FROM ghcr.io/astral-sh/uv:debian-slim

# Upgrade pip and install YOLOv10 version of ultralytics and other common packages

# Set the working directory
WORKDIR /app

# Copy all files into the container
COPY . .

# Install git
RUN apt update && apt install -y git

# Install dependencies listed in pyproject.toml or requirements.lock
RUN uv sync

# Install system dependencies required by YOLOv10 and OpenCV
RUN apt update && apt install -y \
    libgl1 \
    libglib2.0-0 \
    git \
    build-essential \
    libsm6 \
    libxext6 \
    libxrender-dev && \
    rm -rf /var/lib/apt/lists/*

# (Optional) Copy requirements file and install additional Python packages
# COPY requirements.txt .
# RUN uv add -r requirements.txt

# Create the static directory (if not exists)
RUN mkdir -p /app/static

# Run the Flask app
CMD ["uv", "run", "main.py"]
