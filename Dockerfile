# Virgo 1.0 API server. CPU build by default; for an NVIDIA GPU, start FROM a CUDA PyTorch image.
FROM python:3.11-slim
RUN apt-get update && apt-get install -y --no-install-recommends ffmpeg espeak-ng && rm -rf /var/lib/apt/lists/*
WORKDIR /virgo
COPY requirements.txt .
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && pip install --no-cache-dir -r requirements.txt
COPY . .
ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn serve.app:app --host 0.0.0.0 --port ${PORT}"]
