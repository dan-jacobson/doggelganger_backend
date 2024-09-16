# Pull python image, install uv
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Set port
ENV PORT=8000

# Copy requirements file
COPY doggelganger/requirements.txt .

# Install dependencies using uv
RUN uv pip install -r requirements.txt --no-cache-dir --system && rm requirements.txt

# Copy files
COPY doggelganger/serve.py doggelganger/utils.py ./
RUN python -c "from utils import get_model; get_model()"

# Override uv image entrypoint
ENTRYPOINT []

# Run the Litestar application
CMD exec uvicorn serve:app --host 0.0.0.0 --port $PORT
