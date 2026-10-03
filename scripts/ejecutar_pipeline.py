#!/usr/bin/env python
"""Ejecuta el pipeline de ObservaCompras IA paso a paso o completo.

Uso:
    python scripts/ejecutar_pipeline.py todo                 # pipeline completo (2023 y 2024)
    python scripts/ejecutar_pipeline.py ingesta --origen data/raw
    python scripts/ejecutar_pipeline.py perfil
    python scripts/ejecutar_pipeline.py autoencoder
    python scripts/ejecutar_pipeline.py clasificador
    python scripts/ejecutar_pipeline.py app
    python scripts/ejecutar_pipeline.py reporte

Opciones:
    --modo demo     usa un solo semestre: 1er trimestre = "2023", 2º trimestre = "2024"
    --rapido        una sola combinación de hiperparámetros (solo para probar el código)
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from observacompras import config as C  # noqa: E402

PASOS = ["ingesta", "perfil", "autoencoder", "clasificador", "app", "reporte"]


def correr(paso, args):
    t = time.time()
    print(f"\n=== {paso} ===", flush=True)
    if paso == "ingesta":
        from observacompras import ingesta
        r = ingesta.ejecutar(Path(args.origen), modo=args.modo)
    elif paso == "perfil":
        from observacompras import perfil
        r = f"{len(perfil.ejecutar())} perfiles organismo-año"
    elif paso == "autoencoder":
        from observacompras import pipeline_ae
        r = pipeline_ae.ejecutar(rapido=args.rapido)
    elif paso == "clasificador":
        from observacompras import pipeline_clf
        r = pipeline_clf.ejecutar(rapido=args.rapido)
    elif paso == "app":
        from observacompras import pipeline_app
        r = f"{pipeline_app.ejecutar()} OC clasificadas para la vista de detalle"
    elif paso == "reporte":
        from observacompras import reporte
        reporte.generar()
        r = f"reporte en {C.DIR_REPORTES / 'reporte_evaluacion.md'}"
    print(json.dumps(r, ensure_ascii=False, indent=2, default=str) if isinstance(r, dict) else r)
    print(f"({time.time() - t:.0f} s)", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("paso", choices=PASOS + ["todo"])
    ap.add_argument("--origen", default=str(C.DIR_RAW), help="carpeta con los .7z/.zip/.csv descargados")
    ap.add_argument("--modo", choices=["anio", "demo"], default="anio")
    ap.add_argument("--rapido", action="store_true")
    args = ap.parse_args()
    for p in (PASOS if args.paso == "todo" else [args.paso]):
        correr(p, args)


if __name__ == "__main__":
    main()
