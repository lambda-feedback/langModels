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

# Bake the pre-trained Hugging Face weights (GPT-2 and SmolLM2, ~540MB each in
# float32) into the image so gpt2_infer and smollm2_infer need no network
# access at runtime. HF_HOME points any runtime fallback download at /tmp, the
# only writable path on Lambda.
ENV GPT2_MODEL_DIR=/app/hf_models/gpt2 \
    SMOLLM2_MODEL_DIR=/app/hf_models/smollm2-135m-instruct \
    HF_HOME=/tmp/hf
COPY scripts/download_hf_model.py /tmp/download_hf_model.py
# safetensors >=0.8 writes model.safetensors via a tempfile+rename, leaving it
# mode 0600 (root-only). Lambda runs as a non-root user, so open() fails (and
# safetensors reports that as "No such file or directory"). Make it readable.
# One layer per model, so changing one does not re-download the other.
RUN HF_HOME=/tmp/hf_build python /tmp/download_hf_model.py "${GPT2_MODEL_DIR}" --model-id gpt2 && \
    chmod -R a+rX "${GPT2_MODEL_DIR}" && \
    rm -rf /tmp/hf_build
RUN HF_HOME=/tmp/hf_build python /tmp/download_hf_model.py "${SMOLLM2_MODEL_DIR}" \
        --model-id HuggingFaceTB/SmolLM2-135M-Instruct && \
    chmod -R a+rX "${SMOLLM2_MODEL_DIR}" && \
    rm -rf /tmp/hf_build /tmp/download_hf_model.py

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

# Use an IPC socket, not stdio. With stdio the RPC frames share stdout with the
# model code's print() calls; once Python's stdout buffer flushes that text
# into the stream, shimmy loses framing and every later request times out
# (FUNCTION_TIMEOUT after FUNCTION_WORKER_SEND_TIMEOUT). The worker starts in
# ~0.3 s because models (and torch) are imported lazily, so ipc's wait for the
# socket during Lambda init is not a problem here.
ENV FUNCTION_RPC_TRANSPORT="ipc"

# The worker pulls in torch on its first request; on a small (1024 MB) Lambda
# that cold-start import runs ~30-40s. Give shimmy room to wait for it instead
# of killing the half-booted worker at the 30s default. Keep these below the
# Lambda function timeout (currently 175s).
ENV FUNCTION_WORKER_START_TIMEOUT="150s"
ENV FUNCTION_WORKER_SEND_TIMEOUT="150s"

ENV LOG_LEVEL="debug"
