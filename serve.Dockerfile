# Pull python image, install uv
FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim
# COPY --from=ghcr.io/astral-sh/uv:latest /uv /bin/uv

# Set working directory
WORKDIR /app

# Enable bytecode compilation
ENV UV_COMPILE_BYTECODE=1

# Set port
ENV PORT=8000

# Install dependencies
RUN --mount=type=cache,target=/root/.cache/uv \
    --mount=type=bind,source=uv.lock,target=uv.lock,z \
    --mount=type=bind,source=pyproject.toml,target=pyproject.toml,z \
    uv sync --frozen --no-install-project

# Copy files and build project
ADD . /app
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --frozen

RUN uv pip list | grep doggelganger
RUN ["python", "-c", "from doggelganger.utils import download_model_weights; download_model_weights()"]

EXPOSE $PORT

# Override base image entrypoint
ENTRYPOINT []

# Run the Litestar application
CMD exec uvicorn serve:app --host 0.0.0.0 --port $PORT
