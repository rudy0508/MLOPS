"""
evaluate/run.py — Etapa 6: Evaluación en test set holdout + quality gate.

Carga el modelo entrenado, evalúa en el test set y registra métricas en MLflow.
Quality gate: el pipeline falla si AUC < auc_threshold.

Ejecutar: python evaluate/run.py
"""
import argparse
import json
import logging
import pickle
import sys
from pathlib import Path

import mlflow
import pandas as pd
from sklearn.metrics import (
    roc_auc_score, f1_score, recall_score,
    precision_score, accuracy_score, classification_report,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | EVALUATE | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET   = "default"
FEATURES = ["edad", "ingreso_anual", "saldo_cuenta", "meses_empleado",
            "ratio_deuda", "historial_pagos", "productos_banco", "tiene_tarjeta"]
ARTIFACTS = Path("artifacts")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_data",     default="data/credit_test.csv")
    parser.add_argument("--model_path",    default="artifacts/modelo_credit.pkl")
    #parser.add_argument("--mlflow_uri",    default="mlruns")
    parser.add_argument("--mlflow_uri", default=None)
    parser.add_argument("--experiment_name", default="credit_risk")
    parser.add_argument("--auc_threshold", type=float, default=0.80) # Cambiar para que pase a prod
    parser.add_argument("--f1_threshold",  type=float, default=0.70) # Cambiar para que pase a prod
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    for p in [args.test_data, args.model_path]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos anteriores.")

    log.info("Cargando modelo: %s", args.model_path)
    with open(args.model_path, "rb") as f:
        modelo = pickle.load(f)

    log.info("Cargando test set: %s", args.test_data)
    df_test = pd.read_csv(args.test_data)
    X_test  = df_test[FEATURES]
    y_test  = df_test[TARGET]

    y_pred  = modelo.predict(X_test)
    y_proba = modelo.predict_proba(X_test)[:, 1]

    metricas = {
        "test_auc":       round(roc_auc_score(y_test, y_proba), 4),
        "test_f1":        round(f1_score(y_test, y_pred, zero_division=0), 4),
        "test_recall":    round(recall_score(y_test, y_pred, zero_division=0), 4),
        "test_precision": round(precision_score(y_test, y_pred, zero_division=0), 4),
        "test_accuracy":  round(accuracy_score(y_test, y_pred), 4),
    }

    print("\n" + "="*50)
    print(" MÉTRICAS DE EVALUACIÓN (test set holdout)")
    print("="*50)
    for k, v in metricas.items():
        print(f"  {k:<20}: {v:.4f}")
    print("\n" + classification_report(y_test, y_pred))

    # Registrar en MLflow
    #mlflow.set_tracking_uri(args.mlflow_uri)
    #mlflow.set_experiment(args.experiment_name)
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name="evaluate"):
        mlflow.log_metrics(metricas)
        mlflow.log_param("auc_threshold", args.auc_threshold)

    # Guardar métricas JSON
    ARTIFACTS.mkdir(exist_ok=True)
    with open(ARTIFACTS / "eval_metrics.json", "w") as f:
        json.dump(metricas, f, indent=2)
    log.info("Métricas guardadas: artifacts/eval_metrics.json")

    # Quality gate
    print("\n" + "="*50)
    print(" QUALITY GATE")
    print("="*50)
    print(f"  AUC    : {metricas['test_auc']:.4f} (umbral: >= {args.auc_threshold})")
    print(f"  F1     : {metricas['test_f1']:.4f} (umbral: >= {args.f1_threshold})")

    fallos = []
    if metricas["test_auc"] < args.auc_threshold:
        fallos.append(f"AUC {metricas['test_auc']:.4f} < {args.auc_threshold}")
    if metricas["test_f1"] < args.f1_threshold:
        fallos.append(f"F1 {metricas['test_f1']:.4f} < {args.f1_threshold}")

    if fallos:
        log.error("QUALITY GATE FALLIDO: %s", " | ".join(fallos))
        log.error("El pipeline CI/CD se detiene. No se desplegará el modelo.")
        sys.exit(1)

    print("\n  ✓ APROBADO — modelo listo para despliegue")
