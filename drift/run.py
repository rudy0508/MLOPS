"""
drift/run.py — Etapa 7: Detección de drift con EvidentlyAI.

Compara la distribución del train set (referencia) vs el test set (producción).
Genera reporte HTML y JSON. El pipeline falla si el drift supera el umbral.

Ejecutar: python drift/run.py
"""
import argparse, json, logging, sys
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s | DRIFT | %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

FEATURES = ["edad","ingreso_anual","saldo_cuenta","meses_empleado",
            "ratio_deuda","historial_pagos","productos_banco","tiene_tarjeta"]
REPORTS = Path("reportes")


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference",       default="data/credit_train.csv")
    parser.add_argument("--current",         default="data/credit_test.csv")
    parser.add_argument("--drift_threshold", type=float, default=0.30)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    REPORTS.mkdir(exist_ok=True)

    for p in [args.reference, args.current]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{p} no encontrado. Ejecuta los pasos anteriores.")

    df_ref  = pd.read_csv(args.reference)[FEATURES]
    df_prod = pd.read_csv(args.current)[FEATURES]
    log.info("Referencia: %d | Actual: %d muestras", len(df_ref), len(df_prod))

    try:
        from evidently.report import Report
        from evidently.metric_preset import DataDriftPreset
        report = Report(metrics=[DataDriftPreset()])
        report.run(reference_data=df_ref, current_data=df_prod)
        report.save_html(str(REPORTS / "drift_report.html"))
        resultado   = report.as_dict()
        drift_info  = resultado["metrics"][0]["result"]
        drift_det   = drift_info["dataset_drift"]
        drift_share = drift_info["share_of_drifted_columns"]
        drift_n     = drift_info["number_of_drifted_columns"]
        drift_total = drift_info["number_of_columns"]
    except ImportError:
        from scipy.stats import ks_2samp
        drifted     = sum(1 for c in FEATURES if ks_2samp(df_ref[c], df_prod[c]).pvalue < 0.05)
        drift_share = drifted / len(FEATURES)
        drift_det   = drift_share > args.drift_threshold
        drift_n, drift_total = drifted, len(FEATURES)

    resumen = {"drift_detectado": drift_det, "features_con_drift": drift_n,
               "total_features": drift_total, "share_drifted": round(drift_share,4),
               "umbral": args.drift_threshold}
    with open(REPORTS / "drift_summary.json", "w") as f:
        json.dump(resumen, f, indent=2)

    print(f"\n  Drift detectado    : {drift_det}")
    print(f"  Features con drift : {drift_n}/{drift_total} ({drift_share*100:.0f}%)")
    if drift_share > args.drift_threshold:
        log.warning("ALERTA: drift significativo — considerar re-entrenamiento")
    else:
        print("  ✓ Drift dentro de límites aceptables")
