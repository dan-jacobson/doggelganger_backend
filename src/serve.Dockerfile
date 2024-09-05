# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Copy requirements file
COPY src/requirements.txt .

# Install dependencies using uv
RUN uv pip install -r requirements.txt --no-cache-dir --system && rm requirements.txt

# Copy model weights
COPY weights/dinov2-small /app/weights/dinov2-small

# Copy application files
COPY src/serve.py src/utils.py ./

# Override uv image entrypoint
ENTRYPOINT []

# Run the Litestar application
CMD ["uvicorn", "serve:app", "--host", "0.0.0.0", "--port", "80"]
