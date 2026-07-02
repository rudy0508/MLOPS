"""
random_forest/run.py — Etapa 5: Entrenamiento con GridSearchCV + MLflow tracking.

Registra en MLflow: parámetros, métricas CV, modelo como artefacto.
Promueve el mejor modelo al MLflow Model Registry (stage: Staging).

Ejecutar: python random_forest/run.py
"""
import argparse
import json
import logging
import pickle
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import f1_score, roc_auc_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s | TRAIN | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

TARGET   = "default"
FEATURES = ["edad", "ingreso_anual", "saldo_cuenta", "meses_empleado",
            "ratio_deuda", "historial_pagos", "productos_banco", "tiene_tarjeta"]
ARTIFACTS = Path("artifacts")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_data",      default="data/credit_train.csv")
    parser.add_argument("--experiment_name", default="credit_risk")
    #parser.add_argument("--mlflow_uri",      default="mlruns")
    parser.add_argument("--mlflow_uri", default=None)
    parser.add_argument("--model_name",      default="CreditRiskModel")
    parser.add_argument("--n_estimators",    default="100,200,300")
    parser.add_argument("--max_depth",       default="5,8,10")
    parser.add_argument("--min_samples_split", default="2,5,10")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    ARTIFACTS.mkdir(exist_ok=True)

    if not Path(args.train_data).exists():
        raise FileNotFoundError(f"{args.train_data} no encontrado. Ejecuta los pasos anteriores.")

    log.info("Cargando datos de entrenamiento: %s", args.train_data)
    df    = pd.read_csv(args.train_data)
    X     = df[FEATURES]
    y     = df[TARGET]

    #mlflow.set_tracking_uri(args.mlflow_uri)
    #mlflow.set_experiment(args.experiment_name)
    if args.mlflow_uri:
        mlflow.set_tracking_uri(args.mlflow_uri)
    else:
        mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment(args.experiment_name)
    

    param_grid = {
        "n_estimators":      [int(x) for x in args.n_estimators.split(",")],
        "max_depth":         [int(x) for x in args.max_depth.split(",")],
        "min_samples_split": [int(x) for x in args.min_samples_split.split(",")],
    }
    n_combinaciones = len(param_grid["n_estimators"]) * len(param_grid["max_depth"]) * len(param_grid["min_samples_split"])
    log.info("GridSearchCV: %d combinaciones x 5 folds = %d entrenamientos", n_combinaciones, n_combinaciones * 5)

    gs = GridSearchCV(
        RandomForestClassifier(random_state=42, class_weight="balanced"),
        param_grid,
        cv=StratifiedKFold(5, shuffle=True, random_state=42),
        scoring="f1",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )
    gs.fit(X, y)

    log.info("Mejores parámetros: %s", gs.best_params_)
    log.info("Mejor F1 CV:        %.4f", gs.best_score_)

    with mlflow.start_run(run_name="random_forest_gridsearch") as run:
        mlflow.log_params(gs.best_params_)
        mlflow.log_param("n_combinaciones",  n_combinaciones)
        mlflow.log_param("cv_folds",         5)
        mlflow.log_param("scoring",          "f1")
        mlflow.log_param("class_weight",     "balanced")

        y_pred_cv = gs.predict(X)
        mlflow.log_metrics({
            "cv_f1_mean":  round(gs.best_score_, 4),
            "train_f1":    round(f1_score(y, y_pred_cv), 4),
            "train_auc":   round(roc_auc_score(y, gs.predict_proba(X)[:, 1]), 4),
        })

        mlflow.sklearn.log_model(
            gs.best_estimator_,
            artifact_path="random_forest_model",
            registered_model_name=args.model_name,
        )
        run_id = run.info.run_id
        log.info("MLflow Run ID: %s", run_id)

    # Guardar modelo localmente también
    with open(ARTIFACTS / "modelo_credit.pkl", "wb") as f:
        pickle.dump(gs.best_estimator_, f)

    # Guardar run_id para etapas posteriores
    with open(ARTIFACTS / "train_run_id.txt", "w") as f:
        f.write(run_id)

    log.info("Modelo guardado: artifacts/modelo_credit.pkl")
    log.info("Run ID guardado: artifacts/train_run_id.txt")
