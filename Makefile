# Makefile — Pipeline MLOps End-to-End — Riesgo Crediticio
.PHONY: all install lint pipeline etapas serve docker smoke clean help

install:
	pip install -r requirements.txt

all: install lint pipeline serve
	@echo "Pipeline completo OK"

lint:
	flake8 . --max-line-length=100 --exclude=.git,__pycache__,mlruns,artifacts,data,reportes

# Pipeline completo
pipeline:
	python main.py

# Etapas individuales
etapa1: ; python _data/run.py
etapa2: ; python preprocess/run.py
etapa3: ; python segregate/run.py
etapa4: ; pytest check_data/test_data.py -v --tb=short
etapa5: ; python random_forest/run.py
etapa6: ; python evaluate/run.py
etapa7: ; python drift/run.py

# Servidor API
serve:
	uvicorn serve.app:app --host 0.0.0.0 --port 8000 --reload

# Docker
docker-build:
	docker build -t credit-risk-api:local -f serve/Dockerfile .

docker-run:
	docker run -p 8000:8000 --name credit-api credit-risk-api:local

docker-stop:
	docker stop credit-api && docker rm credit-api

# Smoke tests
smoke:
	curl -sf http://localhost:8000/health | python3 -m json.tool
	@echo ""
	curl -X POST http://localhost:8000/predict \
	  -H 'Content-Type: application/json' \
	  -d '{"edad":35,"ingreso_anual":55000,"saldo_cuenta":8000,"meses_empleado":60,"ratio_deuda":0.3,"historial_pagos":2,"productos_banco":3,"tiene_tarjeta":1}' \
	  | python3 -m json.tool

mlflow-ui:
	mlflow ui --host 0.0.0.0 --port 5000

clean:
	rm -rf data/ artifacts/ reportes/ mlruns/ __pycache__ pipeline_run.log
	find . -name "*.pyc" -delete
	@echo "Limpieza completada"

help:
	@echo ""
	@echo "=== Pipeline MLOps End-to-End ==="
	@echo "  make install    — instalar dependencias"
	@echo "  make pipeline   — ejecutar las 7 etapas en orden"
	@echo "  make etapa1-7   — ejecutar una etapa específica"
	@echo "  make serve      — levantar API FastAPI en puerto 8000"
	@echo "  make docker-build — construir imagen Docker"
	@echo "  make docker-run   — ejecutar contenedor"
	@echo "  make smoke      — test rápido de los endpoints"
	@echo "  make mlflow-ui  — UI de MLflow en puerto 5000"
	@echo "  make clean      — limpiar artefactos"
	@echo ""
