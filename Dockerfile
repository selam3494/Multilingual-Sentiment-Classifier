# Dockerfile
FROM python:3.10-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
# Example: run training (override in docker run if you want)
CMD ["python", "-m", "sentiment.trainer"]
