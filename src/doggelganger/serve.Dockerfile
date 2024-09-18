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
COPY requirements.txt .

# Install dependencies using uv
RUN uv pip install -r requirements.txt --no-cache-dir --system && rm requirements.txt

# Copy files
COPY serve.py utils.py ./
RUN python -c "from utils import get_model; get_model()"

EXPOSE $PORT

# Override base image entrypoint
ENTRYPOINT []

# Run the Litestar application
CMD exec uvicorn serve:app --host 0.0.0.0 --port $PORT
