FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TOKENIZERS_PARALLELISM=false \
    TRANSFORMERS_NO_TF=1 \
    HF_HUB_DISABLE_TELEMETRY=1

WORKDIR /app

# Install deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy only what the API needs
COPY sentiment/ sentiment/
COPY api/ api/

# (Optional) copy model if you prefer baking it into the image
# COPY model_en_light_best/ model_en_light_best/

EXPOSE 8000

# Default CMD runs the API
CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
