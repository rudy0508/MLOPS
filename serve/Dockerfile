# serve/Dockerfile — Imagen Docker de la API de riesgo crediticio
FROM python:3.10-slim
LABEL description="API Riesgo Crediticio — MLOps Demo Final"
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1 PORT=8000
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends gcc && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY serve/ ./serve/
COPY artifacts/ ./artifacts/
ENV MODEL_NAME=CreditRiskModel MODEL_STAGE=Production
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=10s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')"
CMD ["uvicorn", "serve.app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
