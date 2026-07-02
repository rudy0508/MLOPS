"""serve/app.py — API REST del modelo de riesgo crediticio."""
import logging, os, pickle
from contextlib import asynccontextmanager
from pathlib import Path
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

logging.basicConfig(level=logging.INFO, format="%(asctime)s | API | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURES = ["edad","ingreso_anual","saldo_cuenta","meses_empleado",
            "ratio_deuda","historial_pagos","productos_banco","tiene_tarjeta"]
UMBRAL, MODEL_PATH = 0.50, Path("artifacts/modelo_credit.pkl")
modelo = None


class SolicitudCredito(BaseModel):
    edad:            int   = Field(..., ge=18, le=100)
    ingreso_anual:   float = Field(..., gt=0)
    saldo_cuenta:    float
    meses_empleado:  int   = Field(..., ge=0)
    ratio_deuda:     float = Field(..., ge=0.0, le=1.0)
    historial_pagos: int   = Field(..., ge=0,  le=2)
    productos_banco: int   = Field(..., ge=1,  le=10)
    tiene_tarjeta:   int   = Field(..., ge=0,  le=1)
    model_config = {"json_schema_extra": {"example": {
        "edad":35,"ingreso_anual":55000.0,"saldo_cuenta":8000.0,
        "meses_empleado":60,"ratio_deuda":0.30,"historial_pagos":2,
        "productos_banco":3,"tiene_tarjeta":1}}}


class PrediccionCredito(BaseModel):
    probabilidad_default: float
    decision:             str
    score:                float
    modelo:               str


class HealthResponse(BaseModel):
    status: str
    modelo: str
    version: str


def cargar_modelo():
    global modelo
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "")
    if mlflow_uri:
        try:
            import mlflow
            mlflow.set_tracking_uri(mlflow_uri)
            modelo = mlflow.sklearn.load_model(
                f"models:/{os.getenv('MODEL_NAME','CreditRiskModel')}/{os.getenv('MODEL_STAGE','Production')}")
            log.info("Modelo desde MLflow Registry: %s", type(modelo).__name__)
            return
        except Exception as e:
            log.warning("MLflow Registry no disponible: %s", e)
    if MODEL_PATH.exists():
        with open(MODEL_PATH, "rb") as f:
            modelo = pickle.load(f)
        log.info("Modelo desde: %s (%s)", MODEL_PATH, type(modelo).__name__)
    else:
        raise FileNotFoundError("Ejecuta: python random_forest/run.py")


@asynccontextmanager
async def lifespan(app: FastAPI):
    cargar_modelo()
    yield


app = FastAPI(title="API Riesgo Crediticio — MLOps Demo", version="1.0.0", lifespan=lifespan)
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])


@app.get("/", tags=["Info"])
def root():
    return {"api":"Riesgo Crediticio","version":"1.0.0","docs":"/docs","health":"/health"}


@app.get("/health", response_model=HealthResponse, tags=["Salud"])
def health():
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    return HealthResponse(status="ok", modelo=type(modelo).__name__, version="1.0.0")


@app.post("/predict", response_model=PrediccionCredito, tags=["Prediccion"])
def predict(solicitud: SolicitudCredito):
    """Evalúa riesgo de default. decision: APROBAR | RECHAZAR"""
    if modelo is None:
        raise HTTPException(status_code=503, detail="Modelo no cargado")
    try:
        df   = pd.DataFrame([solicitud.model_dump()])
        prob = float(modelo.predict_proba(df[FEATURES])[0][1])
        return PrediccionCredito(
            probabilidad_default=round(prob,4),
            decision="RECHAZAR" if prob >= UMBRAL else "APROBAR",
            score=round(prob,4),
            modelo=type(modelo).__name__,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
