# Estado del proyecto y próximos pasos

_Última actualización: 3 de octubre de 2026._

## Hecho

- Código completo de F1 a F6, según la propuesta v4 (`src/observacompras/`, `app/streamlit_app.py`).
- Cuatro cuadernos de Colab (`notebooks/`), generados con `scripts/generar_notebooks.py`.
- 10 pruebas unitarias aprobadas (`python -m pytest tests`).
- Pipeline ejecutado en **modo demostración** con el 1er semestre de 2023 (ene–mar = "2023", abr–jun = "2024").
  Los datos procesados, los modelos y los reportes de esa corrida están en el repositorio.
- App revisada con capturas de sus tres vistas, incluido el flujo de uso de la Sección 6.

## Resultados preliminares (modo demostración)

| Métrica | Resultado | Meta |
|---|---|---|
| Recall@5% de anomalías sintéticas (H1) | 75% (regla z: 45%) | ≥ 80% |
| Estabilidad del top 10 (H1) | 5,4 de 10 | ≥ 7 |
| F1-macro / top-3 del clasificador (H2) | 0,41 / 70%, con codificador de prueba: **no válido** | +5 pts sobre la línea base / ≥ 90% |

Hiperparámetros elegidos para el autoencoder: lr 0,01, batch 8, 76 epochs en el modelo final.

## Pendiente

1. **Datos completos.** Descargar el 2º semestre de 2023 y todo 2024 (OC y licitaciones, sector Salud) desde
   datos-abiertos.chilecompra.cl. Dejarlos **sin descomprimir** en `MyDrive/observacompras-ia/data/raw/`,
   junto al 1er semestre de 2023 (`Ordenes_de_compra_1er_semestre_2023.7z`, `Licitaciones_1er_semestre_2023.7z`).
2. **Colab.** Correr los cuadernos 01 → 04 con `MODO = "anio"`:
   - el 02 tarda unos 45–60 minutos;
   - el 03 conviene correrlo con GPU, porque descarga y usa Sentence-BERT.
   Al terminar, hacer `git push` de `data/processed`, `models` y `reports`.
3. **H3.** Los autores completan `reports/revision_explicabilidad_H3.csv` con los 10 organismos más atípicos.
4. **H4.** Pruebas con 5 usuarios según `docs/guia_pruebas_usuario.md`; los resultados van en `reports/pruebas_usuario_H4.csv`.
5. Volver a correr el cuaderno 04 para actualizar `reports/reporte_evaluacion.md`.
6. Preparar la presentación del prospecto y el video de demostración de 5 minutos.

## Para retomar con Claude

Abrir una conversación nueva y pegar el enlace del repositorio, `https://github.com/lcornejom/observacompras-ia`.
Pedir que lea este archivo y `docs/decisiones.md`, y adjuntar los archivos de datos que se quieran procesar.
