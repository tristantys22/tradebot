FROM python:3.11-slim
ENV PIP_NO_CACHE_DIR=1 PYTHONUNBUFFERED=1 HF_HUB_DISABLE_PROGRESS_BARS=1
RUN apt-get update && apt-get install -y --no-install-recommends git && rm -rf /var/lib/apt/lists/*
RUN pip install torch --index-url https://download.pytorch.org/whl/cpu
RUN pip install chronos-forecasting einops huggingface_hub safetensors tqdm matplotlib pandas numpy yfinance
RUN git clone --depth 1 https://github.com/shiyu-coder/Kronos /kronos
RUN python -c "from huggingface_hub import snapshot_download as d; [d(r) for r in ['NeoQuasar/Kronos-small','NeoQuasar/Kronos-Tokenizer-base','amazon/chronos-bolt-small','amazon/chronos-bolt-base']]"
COPY fm.py /fm.py
CMD ["python", "/fm.py"]
