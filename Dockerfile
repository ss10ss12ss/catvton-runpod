FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/tmp/hf \
    HF_HUB_DISABLE_XET=1

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git libgl1-mesa-glx libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Clone CatVTON source
RUN git clone https://github.com/Zheng-Chong/CatVTON.git /app/CatVTON

# Install Python dependencies (pin huggingface_hub to avoid httpx.Client token error)
COPY requirements.txt /app/requirements.txt
# Force downgrade huggingface_hub first (base image ships v1.x which breaks diffusers 0.27)
RUN pip install --no-cache-dir --force-reinstall huggingface_hub==0.23.4 \
    && pip install --no-cache-dir --force-reinstall -r /app/requirements.txt

# Pre-download models (baked into image for fast cold start)
RUN python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('booksforcharlie/stable-diffusion-inpainting'); \
    snapshot_download('stabilityai/sd-vae-ft-mse'); \
    snapshot_download('zhengchong/CatVTON')"

COPY handler.py /app/handler.py

ENV CATVTON_SRC=/app/CatVTON
ENV BASE_MODEL_PATH=booksforcharlie/stable-diffusion-inpainting
ENV ATTN_REPO=zhengchong/CatVTON
ENV ATTN_VERSION=mix

CMD ["python", "-u", "/app/handler.py"]
