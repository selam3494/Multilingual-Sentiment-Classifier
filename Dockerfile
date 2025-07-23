# Dockerfile (for training/eval container)
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# default: train (override with docker run CMD)
CMD ["python", "-m", "sentiment.trainer"]
