FROM python:3.10-slim
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY . .
# Usage: docker run -v $(pwd)/data:/app/data -v $(pwd)/outputs:/app/outputs <image> --source data/metro.mp4
ENTRYPOINT ["python", "main.py"]
