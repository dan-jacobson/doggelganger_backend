# Use a Python image with uv pre-installed
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV PORT 8000

# Copy requirements file
COPY src/requirements.txt .

# Install dependencies using uv
RUN uv pip install -r requirements.txt --no-cache-dir --system && rm requirements.txt

# Copy files
COPY src/serve.py src/utils.py ./
RUN python -c "from utils import get_model; get_model()"

# Override uv image entrypoint
ENTRYPOINT []

# Run the Litestar application
CMD exec uvicorn serve:app --host 0.0.0.0 --port $PORT
