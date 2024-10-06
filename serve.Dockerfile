# Pull python image, install uv
FROM python:3.12-slim-bookworm
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy
ENV UV_PROJECT_ENVIRONMENT="/usr/local/"

# Set working directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev --index-strategy unsafe-best-match

# Copy files and build project
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

# Set HF cache dir and download weights
ENV HF_HOME=.cache/huggingface
RUN python -c "from doggelganger.utils import download_model_weights; download_model_weights()"

# Set port as env var, necessary for Cloud Run as I understand it
ENV PORT=8000
EXPOSE $PORT

# Override base image entrypoint
ENTRYPOINT []

# Run the Litestar application
CMD uv run serve --host 0.0.0.0 --port $PORT