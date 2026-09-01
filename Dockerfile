FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12 AS builder

RUN pip install poetry==2.2.1

ENV POETRY_NO_INTERACTION=1 \
    POETRY_VIRTUALENVS_IN_PROJECT=1 \
    POETRY_VIRTUALENVS_CREATE=1 \
    POETRY_CACHE_DIR=/tmp/poetry_cache

COPY pyproject.toml poetry.lock ./

# Install dependencies and clean up in same layer
RUN --mount=type=cache,target=$POETRY_CACHE_DIR \
    poetry install --without dev --no-root && \
    # Remove unnecessary files from venv to reduce size
    find /app/.venv -name "*.pyc" -delete && \
    find /app/.venv -name "__pycache__" -type d -exec rm -rf {} + && \
    find /app/.venv -name "*.pyo" -delete && \
    # Remove test files and documentation
    find /app/.venv -path "*/tests/*" -delete && \
    find /app/.venv -path "*/test/*" -delete && \
    find /app/.venv -name "*.md" -delete && \
    find /app/.venv -name "*.txt" -delete

FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12

ENV VIRTUAL_ENV=/app/.venv \
    PATH="/app/.venv/bin:$PATH"

# Copy the cleaned virtual environment
COPY --from=builder ${VIRTUAL_ENV} ${VIRTUAL_ENV}

# Copy evaluation function first (smaller, changes more often)
COPY evaluation_function ./evaluation_function

# Precompile python files for faster startup (do this last)
RUN python -m compileall -q .

# Environment variables
# Command to start the evaluation function with
ENV FUNCTION_COMMAND="python"

# Args to start the evaluation function with
ENV FUNCTION_ARGS="-m,evaluation_function.main"

ENV FUNCTION_INTERFACE="rpc"
ENV FUNCTION_RPC_TRANSPORT="stdio"

# The worker pulls in torch on its first request; on a small (1024 MB) Lambda
# that cold-start import runs ~30-40s. Give shimmy room to wait for it instead
# of killing the half-booted worker at the 30s default. Keep these below the
# Lambda function timeout (currently 175s).
ENV FUNCTION_WORKER_START_TIMEOUT="150s"
ENV FUNCTION_WORKER_SEND_TIMEOUT="150s"

ENV LOG_LEVEL="debug"
