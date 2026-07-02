"""
preprocess/run.py — Etapa 2: Limpieza y transformación del dataset.

Operaciones:
  - Imputar nulos (mediana numérico, moda categórico)
  - Clip de outliers en variables numéricas
  - Normalización de variables continuas (MinMaxScaler)
  - Encoding de variables categóricas (si hubiera)
  - Validación de schema mínimo post-limpieza

Ejecutar: python preprocess/run.py
"""
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | PREPROCESS | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURES_NUM = ["edad", "ingreso_anual", "saldo_cuenta", "meses_empleado", "ratio_deuda"]
FEATURES_CAT = ["historial_pagos", "productos_banco", "tiene_tarjeta"]
TARGET       = "default"
ALL_COLS     = FEATURES_NUM + FEATURES_CAT + [TARGET]


def imputar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for col in FEATURES_NUM:
        if col in df.columns:
            n = df[col].isnull().sum()
            if n > 0:
                df[col].fillna(df[col].median(), inplace=True)
                log.info("  Imputados %d nulos en '%s' (mediana)", n, col)
    for col in FEATURES_CAT:
        if col in df.columns:
            n = df[col].isnull().sum()
            if n > 0:
                df[col].fillna(df[col].mode()[0], inplace=True)
                log.info("  Imputados %d nulos en '%s' (moda)", n, col)
    return df


def clip_outliers(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    clips = {
        "ingreso_anual":  (0, 200000),
        "saldo_cuenta":   (-5000, 50000),
        "meses_empleado": (0, 480),
        "ratio_deuda":    (0, 1),
    }
    for col, (lo, hi) in clips.items():
        if col in df.columns:
            n = ((df[col] < lo) | (df[col] > hi)).sum()
            df[col] = df[col].clip(lo, hi)
            log.info("  Clipeados %d outliers en '%s'", n, col)
    return df


def normalizar(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    scaler = MinMaxScaler()
    cols = [c for c in FEATURES_NUM if c in df.columns]
    df[cols] = scaler.fit_transform(df[cols])
    log.info("  Normalización MinMaxScaler aplicada a: %s", cols)
    return df


def validar_schema(df: pd.DataFrame) -> None:
    errores = []
    nulos = df.isnull().sum().sum()
    if nulos > 0:
        errores.append(f"NULOS RESIDUALES: {nulos}")
    missing = set(ALL_COLS) - set(df.columns)
    if missing:
        errores.append(f"COLUMNAS FALTANTES: {missing}")
    if errores:
        raise ValueError("VALIDACIÓN POST-PREPROCESS FALLIDA:\n" + "\n".join(errores))
    log.info("  Schema OK — 0 nulos, todas las columnas presentes")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input",  default="data/credit_data_raw.csv")
    parser.add_argument("--output", default="data/credit_data_clean.csv")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    if not Path(args.input).exists():
        raise FileNotFoundError(f"{args.input} no encontrado. Ejecuta: python _data/run.py")

    log.info("Cargando: %s", args.input)
    df = pd.read_csv(args.input)
    log.info("Shape inicial: %d x %d | Nulos: %d", df.shape[0], df.shape[1], df.isnull().sum().sum())

    df = imputar(df)
    df = clip_outliers(df)
    df = normalizar(df)
    validar_schema(df)

    df.to_csv(args.output, index=False)
    log.info("Dataset limpio guardado: %s (%d filas)", args.output, len(df))
