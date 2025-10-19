FROM ghcr.io/lambda-feedback/evaluation-function-base/python:3.12 AS builder

RUN pip install poetry==1.8.3

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
ENV FUNCTION_COMMAND="python" \
    FUNCTION_ARGS="-m,evaluation_function.main" \
    FUNCTION_RPC_TRANSPORT="ipc" \
    LOG_LEVEL="debug"