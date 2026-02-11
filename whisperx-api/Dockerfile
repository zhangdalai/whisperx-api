FROM nvidia/cuda:12.2.2-cudnn8-runtime-ubuntu22.04

WORKDIR /app

# Allow unsafe torch.load for compatibility with older model checkpoints
ENV TORCH_FORCE_WEIGHTS_ONLY_LOAD=0

COPY . .

RUN export DEBIAN_FRONTEND=noninteractive \
    && apt-get -qq update \
    && apt-get -qq install --no-install-recommends ffmpeg curl git python3 python3-pip \
    && pip install --no-cache --upgrade pip \
    && pip install --no-cache torch==2.6.0 torchaudio==2.6.0 --extra-index-url https://download.pytorch.org/whl/cu122 \
    && pip install --no-cache transformers==4.46.0 \
    && pip install --no-cache -r requirements.txt \
    && pip install --no-cache python-multipart \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /root/.cache/pip/

HEALTHCHECK --interval=30s --timeout=10s --start-period=15s \
    CMD ["python3", "healthcheck.py"]

EXPOSE 8000
CMD ["python3", "main.py"]
