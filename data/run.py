"""
_data/run.py — Etapa 1: Descarga/generación del dataset de riesgo crediticio.

En producción: descarga desde URL o W&B artifact.
En Codespace:  genera un dataset sintético realista con las mismas
               variables que el dataset de crédito real.

Ejecutar: python _data/run.py
"""
import argparse
import logging
import numpy as np
import pandas as pd
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DOWNLOAD | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

DATA_DIR = Path("data")
N        = 15000
SEED     = 42


def generar_dataset_credito(n: int = N, seed: int = SEED) -> pd.DataFrame:
    """Genera dataset sintético de riesgo crediticio (15,000 clientes)."""
    rng = np.random.default_rng(seed)

    edad            = rng.integers(18, 75, n)
    ingreso_anual   = rng.normal(45000, 25000, n).clip(8000, 250000)
    saldo_cuenta    = rng.normal(3000, 8000, n).clip(-5000, 50000)
    meses_empleado  = rng.integers(0, 480, n)
    ratio_deuda     = rng.beta(2, 5, n)
    historial_pagos = rng.choice([0, 1, 2], n, p=[0.15, 0.35, 0.50])
    productos_banco = rng.integers(1, 8, n)
    tiene_tarjeta   = rng.choice([0, 1], n, p=[0.35, 0.65])

    # Target correlacionado con las variables
    score_default = (
        -0.02 * (ingreso_anual / 10000)
        - 0.01 * (saldo_cuenta / 1000)
        + 1.50 * ratio_deuda
        - 0.80 * historial_pagos
        - 0.01 * meses_empleado
        + 0.05 * (25 - edad).clip(0, 25)
        + rng.normal(0, 0.5, n)
    )
    prob_default = 1 / (1 + np.exp(-score_default))
    default      = (rng.uniform(0, 1, n) < prob_default).astype(int)

    df = pd.DataFrame({
        "edad":            edad,
        "ingreso_anual":   ingreso_anual.round(2),
        "saldo_cuenta":    saldo_cuenta.round(2),
        "meses_empleado":  meses_empleado,
        "ratio_deuda":     ratio_deuda.round(4),
        "historial_pagos": historial_pagos,
        "productos_banco": productos_banco,
        "tiene_tarjeta":   tiene_tarjeta,
        "default":         default,
    })

    # Introducir nulos realistas (~2% por columna)
    for col in ["ingreso_anual", "saldo_cuenta", "meses_empleado"]:
        idx = rng.choice(n, size=int(n * 0.02), replace=False)
        df.loc[idx, col] = np.nan

    return df


def parse_args():
    parser = argparse.ArgumentParser(description="Descarga o genera el dataset de crédito")
    parser.add_argument("--output",  default="data/credit_data_raw.csv")
    parser.add_argument("--n_rows",  type=int, default=N)
    parser.add_argument("--seed",    type=int, default=SEED)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    DATA_DIR.mkdir(exist_ok=True)

    log.info("Generando dataset de riesgo crediticio (%d filas)...", args.n_rows)
    df = generar_dataset_credito(n=args.n_rows, seed=args.seed)

    output = Path(args.output)
    df.to_csv(output, index=False)

    log.info("Dataset guardado: %s", output)
    log.info("Shape: %d filas x %d columnas", *df.shape)
    log.info("Tasa de default: %.1f%%", df['default'].mean() * 100)
    log.info("Nulos: %d", df.isnull().sum().sum())
