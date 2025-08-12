FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    curl bash build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh && \
    echo 'export PATH="/root/.cargo/bin:$PATH"' >> ~/.bashrc && \
    . ~/.bashrc

COPY pyproject.toml uv.lock ./

RUN . ~/.bashrc && \
    uv venv && \
    . .venv/bin/activate && \
    uv pip compile pyproject.toml > requirements.txt && \
    uv pip install --requirement requirements.txt

COPY . .

ENV PYTHONUNBUFFERED=1 \
    WORKERS=6 \
    PORT=8000

HEALTHCHECK --interval=30s --timeout=30s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:$PORT/health || exit 1

EXPOSE $PORT

CMD ["sh", "-c", "uvicorn main:app --host 0.0.0.0 --port $PORT --workers $WORKERS"]
