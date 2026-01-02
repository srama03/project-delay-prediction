FROM python:3.9-slim

# Set working directory inside container
WORKDIR /app

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Copy requirements first (better caching)
COPY requirements.txt /app/requirements.txt

# install git 
RUN apt-get update && apt-get install -y git && rm -rf /var/lib/apt/lists/*

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the project
COPY . /app

# Default command (can be overridden)
CMD ["python", "-m", "src.infer", "--input", "examples/ex1.json"]
