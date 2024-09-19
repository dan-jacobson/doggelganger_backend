# Pull python image, install uv
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Set working directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock,z \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,z \
    uv sync --frozen --no-install-project

# Copy files and build project
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

# Set HF cache dir and download weights
ENV HF_HOME=.cache/huggingface
RUN ["uv", "run", "python", "-c", "from doggelganger.utils import download_model_weights; download_model_weights()"]

# Set port as env var, necessary for Cloud Run as I understand it
ENV PORT=8000
EXPOSE $PORT

# Override base image entrypoint
ENTRYPOINT []

# Run the Litestar application
CMD uv run uvicorn doggelganger.serve:app --host 0.0.0.0 --port $PORT
