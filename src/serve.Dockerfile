# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Copy requirements file
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install -r requirements.txt

# Copy model weights
COPY weights/dinov2-small /app/weights/dinov2-small

# Copy application files
COPY serve.py utils.py ./

# Run the FastAPI application
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "8000"]
