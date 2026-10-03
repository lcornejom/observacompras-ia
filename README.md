# ObservaCompras IA

**Detección de patrones atípicos y clasificación de licitaciones en las compras públicas de Chile
mediante aprendizaje profundo.**

MVP de la asignatura *Aplicaciones de Inteligencia Artificial (MTI-493)*, Magíster en Tecnologías de
la Información, Universidad Técnica Federico Santa María.
Profesor: Sr. Werner Creixell · Alumnos: Alejandra Cornejo Jara y Leoncio Cornejo Miranda.

> ⚠️ Un patrón atípico no es evidencia de irregularidad; indica dónde conviene mirar.

## Qué hace

Dos redes neuronales en Keras, entrenadas con datos abiertos de ChileCompra (sector Salud, 2023–2024),
presentadas en una aplicación Streamlit para un analista no técnico:

| ID | Funcionalidad | Implementación |
|---|---|---|
| F1 | Ingesta y limpieza | [`ingesta.py`](src/observacompras/ingesta.py) |
| F2 | Perfil de compra (10 variables por organismo y año) | [`perfil.py`](src/observacompras/perfil.py) |
| F3 | Detector de patrones atípicos: autoencoder (no supervisado) | [`autoencoder.py`](src/observacompras/autoencoder.py), [`pipeline_ae.py`](src/observacompras/pipeline_ae.py) |
| F4 | Clasificador UNSPSC: Sentence-BERT congelado + red densa (transfer learning) | [`clasificador.py`](src/observacompras/clasificador.py), [`pipeline_clf.py`](src/observacompras/pipeline_clf.py) |
| F5 | Búsqueda semántica (similitud coseno de los mismos embeddings) | [`busqueda.py`](src/observacompras/busqueda.py) |
| F6 | Aplicación web: ranking con explicación, clasificador y buscador | [`app/streamlit_app.py`](app/streamlit_app.py) |

La evaluación de las hipótesis H1–H4 (Tabla 4 del documento) está en
[`reports/reporte_evaluacion.md`](reports/reporte_evaluacion.md). Las decisiones que el documento deja
abiertas están justificadas en [`docs/decisiones.md`](docs/decisiones.md).

## Estado actual

| Componente | Estado |
|---|---|
| Código F1–F6, cuadernos, pruebas unitarias | ✅ Completo |
| Datos y modelos incluidos en el repositorio | 🟡 **Modo demostración** con el 1er semestre de 2023 (ene–mar = "2023", abr–jun = "2024") |
| Embeddings del clasificador | 🟡 Codificador de prueba (TF-IDF + SVD); falta generar los de Sentence-BERT en Colab (cuaderno 03) |
| Ejecución con 2023 y 2024 completos | ⏳ Pendiente: requiere los archivos de ChileCompra de ambos años |
| H3 (revisión de explicaciones) y H4 (pruebas con usuarios) | ⏳ Pendiente: plantillas en `reports/` y guía en `docs/guia_pruebas_usuario.md` |

Resultados preliminares (modo demostración): el autoencoder ubica el **80%** de las anomalías
sintéticas en el 5% superior, frente al 45% de la regla z (H1 cumple), pero la estabilidad del top 10
entre folds es 5,4 de 10 (meta: 7). Las métricas del clasificador no son válidas hasta generar los
embeddings con Sentence-BERT.

## Estructura

```
├── app/streamlit_app.py          Aplicación (F6)
├── notebooks/                    Cuadernos de Google Colab
│   ├── 01_ingesta_y_perfiles     F1, F2
│   ├── 02_autoencoder            F3, H1 y curvas de pérdida
│   ├── 03_clasificador_unspsc    F4, F5, H2 y curvas de pérdida
│   └── 04_evaluacion             Reporte de las seis métricas
├── src/observacompras/           Código fuente (config.py concentra todos los parámetros)
├── scripts/ejecutar_pipeline.py  Pipeline por línea de comandos
├── data/raw/                     Archivos descargados de ChileCompra (no versionados)
├── data/processed/               Datos procesados en Parquet y embeddings
├── models/                       Modelos entrenados (.keras) y escaladores
├── reports/                      Reporte de evaluación, figuras y planillas de revisión
├── docs/                         Decisiones de implementación y guía de pruebas con usuarios
└── tests/                        Pruebas unitarias (pytest)
```

## Cómo ejecutarlo

### 1. Ver la aplicación (computador personal)

El repositorio incluye los datos procesados y los modelos entrenados, así que la aplicación funciona sin
volver a entrenar:

```bash
git clone https://github.com/lcornejom/observacompras-ia.git
cd observacompras-ia
python -m venv .venv && source .venv/bin/activate      # en Windows: .venv\Scripts\activate
pip install -r requirements.txt
streamlit run app/streamlit_app.py
```

La primera vez que se escribe un texto libre en el clasificador o el buscador se descarga
Sentence-BERT (~470 MB). Las demás funciones no lo necesitan.

### 2. Reproducir el entrenamiento (Google Colab)

1. Descargue desde [datos-abiertos.chilecompra.cl](https://datos-abiertos.chilecompra.cl) los archivos
   del sector Salud de 2023 y 2024 (órdenes de compra y licitaciones).
2. Abra `notebooks/01_ingesta_y_perfiles.ipynb` en Colab. La primera celda monta Google Drive y clona el
   repositorio en `MyDrive/observacompras-ia`; copie los archivos descargados en su carpeta `data/raw/`.
3. Ejecute los cuadernos 01 a 04 en orden (para el 03 conviene activar la GPU).

### 3. Reproducir por línea de comandos

```bash
python scripts/ejecutar_pipeline.py todo --origen data/raw
# o paso a paso: ingesta | perfil | autoencoder | clasificador | app | reporte
python -m pytest tests                    # pruebas unitarias
```

`--modo demo` ejecuta todo con un solo semestre (1er trimestre = "2023", 2º trimestre = "2024"), útil
para validar el código.

## Modelos

**Autoencoder (F3).** Entrada: 10 variables escaladas a [0, 1]. Dense 16 → 4 (latente) → 16 → 10, ReLU y
sigmoide, pérdida `mse`, RMSprop. Validación K-fold (K = 5) con early stopping; búsqueda de
learning rate {1e-2, 1e-3, 1e-4} × batch size {8, 16, 32}. Puntaje = error cuadrático medio de
reconstrucción; el error de cada variable es la explicación.

**Clasificador (F4).** `paraphrase-multilingual-MiniLM-L12-v2` congelado produce 384 dimensiones por
licitación; encima, Dense 128 (ReLU) → Dropout 0,3 → Dense n_clases (softmax), pérdida
`categorical_crossentropy`, Adam. Entrenamiento 85/15 sobre 2023 y prueba con 2024; búsqueda de
learning rate {1e-2, 1e-3, 1e-4} × batch size {64, 128, 256}. Línea base: TF-IDF + regresión logística.

## IA responsable

* El sistema señala dónde mirar; no califica conductas. Cada vista muestra una advertencia fija, el
  periodo y los tipos de compra incluidos.
* Los proveedores que son personas naturales aparecen anonimizados (Ley 19.628 y Ley 21.719).
* El código, los datos procesados y las semillas (42) están publicados para reproducir los resultados.

## Uso de inteligencia artificial

El código se desarrolló con apoyo de un asistente de IA generativa (Claude, de Anthropic), a partir del
diseño definido por los autores en el documento de propuesta. Los autores revisaron el código y los
resultados y asumen la responsabilidad por ellos.

## Fuente de datos

ChileCompra — Dirección de Compras y Contratación Pública, *Datos abiertos de compras públicas*,
https://datos-abiertos.chilecompra.cl
