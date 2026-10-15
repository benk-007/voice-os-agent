# NOTE: not tested yet (never built). Feedback / fixes welcome.
# Text-mode agent (no microphone / GUI): CPU-only PyTorch.
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 TOKENIZERS_PARALLELISM=false \
    VOA_WORKSPACE=/workspace HF_HOME=/cache/huggingface

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --extra-index-url https://download.pytorch.org/whl/cpu -r requirements.txt

COPY pyproject.toml README.md ./
COPY src ./src
COPY evaluation ./evaluation
COPY data ./data
COPY tests ./tests
COPY _archive/original ./_archive/original
RUN pip install --no-cache-dir --no-deps -e .

# Non-root user; the agent can only touch /workspace.
RUN useradd --create-home agent && mkdir -p /workspace /cache && chown agent /workspace /cache
USER agent

ENTRYPOINT ["python", "-m", "voice_os_agent"]
CMD ["--help"]
