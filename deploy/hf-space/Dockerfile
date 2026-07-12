# m.ai on Hugging Face Spaces (Docker SDK). Build context = repo root.
#   docker build -f deploy/hf-space/Dockerfile -t mai .
FROM python:3.11-slim

WORKDIR /app

# CPU-only torch (a fraction of the CUDA wheel's size) + tiktoken
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu \
 && pip install --no-cache-dir tiktoken

# pre-warm the gpt2 BPE vocab into the image so runtime needs no network
ENV TIKTOKEN_CACHE_DIR=/tmp/tiktoken
RUN mkdir -p /tmp/tiktoken && python -c "import tiktoken; tiktoken.get_encoding('gpt2')"

# app code + whatever checkpoints are in the build context
COPY . /app

# HF routes external traffic to app_port (declared in README frontmatter)
EXPOSE 7860
ENV MAI_HOST=0.0.0.0
CMD ["python", "mai_serve.py", "--host", "0.0.0.0", "--port", "7860"]
