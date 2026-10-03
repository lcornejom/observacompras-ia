"""Figuras del reporte de evaluación (matplotlib, sin estilo dependiente de tema)."""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

AZUL, NARANJO, GRIS = "#2a6fb0", "#d9822b", "#8a8f98"


def _guardar(fig, ruta):
    fig.tight_layout()
    fig.savefig(ruta, dpi=140)
    plt.close(fig)


def curva_perdida(curva: pd.DataFrame, titulo: str, ruta, epoch_optimo: int | None = None):
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(curva["epoch"], curva["loss"], color=AZUL, label="Entrenamiento")
    ax.plot(curva["epoch"], curva["val_loss"], color=NARANJO, label="Validación")
    if epoch_optimo:
        ax.axvline(epoch_optimo, color=GRIS, ls="--", lw=1, label=f"Mejor epoch ({epoch_optimo})")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Pérdida")
    ax.set_title(titulo, fontsize=11)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False)
    _guardar(fig, ruta)


def heatmap_busqueda(tabla: pd.DataFrame, valor: str, titulo: str, ruta):
    piv = tabla.groupby(["learning_rate", "batch_size"])[valor].mean().unstack()
    fig, ax = plt.subplots(figsize=(5, 3.6))
    im = ax.imshow(piv.values, cmap="Blues_r")
    ax.set_xticks(range(piv.shape[1]), piv.columns)
    ax.set_yticks(range(piv.shape[0]), [f"{v:g}" for v in piv.index])
    ax.set_xlabel("Batch size")
    ax.set_ylabel("Learning rate")
    for i in range(piv.shape[0]):
        for j in range(piv.shape[1]):
            ax.text(j, i, f"{piv.values[i, j]:.4g}", ha="center", va="center", fontsize=8)
    ax.set_title(titulo, fontsize=10)
    fig.colorbar(im, ax=ax, shrink=0.8)
    _guardar(fig, ruta)


def recall_por_tipo(sint: pd.DataFrame, ruta):
    r = sint.groupby("tipo_anomalia")[["detectada_ae", "detectada_z"]].mean()
    x = np.arange(len(r))
    fig, ax = plt.subplots(figsize=(7, 3.8))
    ax.bar(x - 0.2, r["detectada_ae"], 0.4, color=AZUL, label="Autoencoder")
    ax.bar(x + 0.2, r["detectada_z"], 0.4, color=GRIS, label="Línea base z")
    ax.set_xticks(x, r.index, rotation=15)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Recall@5%")
    ax.axhline(0.8, color=NARANJO, ls="--", lw=1, label="Meta 80%")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Anomalías sintéticas detectadas en el 5% superior", fontsize=11)
    _guardar(fig, ruta)


def comparacion_clasificadores(metricas: dict, ruta):
    nombres = ["F1-macro", "Exactitud top-3"]
    claves = ["f1_macro", "top3"]
    x = np.arange(len(nombres))
    fig, ax = plt.subplots(figsize=(6, 3.8))
    ax.bar(x - 0.2, [metricas["transfer_learning"][k] for k in claves], 0.4, color=AZUL, label="SBERT + Dense")
    ax.bar(x + 0.2, [metricas["linea_base"][k] for k in claves], 0.4, color=GRIS, label="TF-IDF + Reg. logística")
    ax.set_xticks(x, nombres)
    ax.set_ylim(0, 1.05)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, fontsize=8)
    ax.set_title("Clasificador UNSPSC: prueba con el año de evaluación", fontsize=11)
    _guardar(fig, ruta)
