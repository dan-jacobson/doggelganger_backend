FROM python:3.12-slim-bookworm AS builder
COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Need to explicitly install git for `vecs`
RUN apt-get update && apt-get install -y git

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1
ENV UV_LINK_MODE=copy

# Set working directory
WORKDIR /app

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml \
    uv sync --frozen --no-install-project --no-dev

# Copy files and build project
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen --no-dev

FROM python:3.12-slim-bookworm
WORKDIR /app

COPY --from=builder --chown=app:app /app /app
ENV PATH="/app/.venv/bin:$PATH"

# Set port as env var, necessary for Cloud Run as I understand it
ENV PORT=8000
EXPOSE $PORT

# Set HF cache dir and download weights
ENV HF_HOME=/app/.cache/huggingface
RUN /app/.venv/bin/python -c "from doggelganger.utils import download_model_weights; download_model_weights()"

# Let's us easily override at runtime with other uvicorn args, like --log-level
RUN chmod +x /app/entrypoint.sh
ENTRYPOINT [ "/app/entrypoint.sh" ]
CMD []