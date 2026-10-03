"""Genera los cuadernos de Google Colab en notebooks/ (python scripts/generar_notebooks.py)."""
from pathlib import Path

import nbformat as nbf

RAIZ = Path(__file__).resolve().parents[1]
REPO = "https://github.com/lcornejom/observacompras-ia"

PREPARACION = f'''# Preparación del entorno (Google Colab o local)
import os, sys, subprocess
EN_COLAB = "google.colab" in sys.modules
if EN_COLAB:
    from google.colab import drive
    drive.mount("/content/drive")
    # El repositorio se clona en Drive para que datos y modelos persistan entre cuadernos
    RAIZ = "/content/drive/MyDrive/observacompras-ia"
    if not os.path.exists(RAIZ):
        subprocess.run(["git", "clone", "{REPO}", RAIZ], check=True)
    os.chdir(RAIZ)
    subprocess.run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements-colab.txt"], check=True)
else:
    RAIZ = os.path.abspath("..") if os.path.basename(os.getcwd()) == "notebooks" else os.getcwd()
    os.chdir(RAIZ)
sys.path.insert(0, os.path.join(RAIZ, "src"))
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

# Modo de periodos: "anio" (2023 y 2024 completos, como en el documento) o
# "demo" (solo un semestre: 1er trimestre = "2023", 2º trimestre = "2024")
MODO = "anio"

from observacompras import config as C
print("Raíz del proyecto:", RAIZ)'''


def md(t):
    return nbf.v4.new_markdown_cell(t)


def code(t):
    return nbf.v4.new_code_cell(t)


def guardar(nombre, celdas):
    nb = nbf.v4.new_notebook()
    nb.metadata = {"kernelspec": {"name": "python3", "display_name": "Python 3"},
                   "language_info": {"name": "python"}, "colab": {"provenance": []},
                   "accelerator": "GPU"}
    nb.cells = celdas
    nbf.write(nb, RAIZ / "notebooks" / nombre)


ENCABEZADO = ("**ObservaCompras IA** · MVP · Aplicaciones de Inteligencia Artificial (MTI-493), UTFSM  \n"
              "Profesor: Sr. Werner Creixell · Alumnos: Alejandra Cornejo Jara y Leoncio Cornejo Miranda\n\n")

guardar("01_ingesta_y_perfiles.ipynb", [
    md("# 01 · Ingesta, limpieza y perfil de compra (F1 y F2)\n\n" + ENCABEZADO +
       "Este cuaderno implementa las funcionalidades **F1** (ingesta y limpieza) y **F2** (perfil de compra "
       "con 10 variables por organismo y año) de la Tabla 2 del documento.\n\n"
       "**Datos de entrada.** Descargue desde https://datos-abiertos.chilecompra.cl los archivos del sector "
       "Salud (órdenes de compra y licitaciones) de 2023 y 2024 y copie los `.7z`, `.zip` o `.csv` en "
       "`MyDrive/observacompras-ia/data/raw/`. La ingesta los descomprime y reconoce cada archivo por su encabezado."),
    code(PREPARACION),
    code("import os\nprint(sorted(os.listdir(C.DIR_RAW)))"),
    md("## F1 · Ingesta y limpieza\n\n"
       "Pasos: filtrar el sector Salud y 2023–2024; colapsar las líneas de ítem a una fila por orden de compra; "
       "excluir OC canceladas; validar montos (> 0) y fechas; normalizar RUT de organismos y proveedores; "
       "y anonimizar a los proveedores persona natural (RUT < 50.000.000) con un seudónimo estable, "
       "conforme a la Ley 19.628 y la Ley 21.719 (Tabla 5)."),
    code("from observacompras import ingesta\nimport json\n"
         "resumen = ingesta.ejecutar(C.DIR_RAW, modo=MODO)\n"
         "print(json.dumps(resumen, indent=2, ensure_ascii=False))"),
    code("import pandas as pd\nfrom observacompras.ingesta import leer_oc\noc = leer_oc()\n"
         "lic = pd.read_parquet(C.DIR_PROCESADOS / 'licitaciones.parquet')\n"
         "display(oc.head())\ndisplay(lic.head())"),
    md("## F2 · Perfil de compra\n\n"
       "| Variable | Definición |\n|---|---|\n"
       "| N.º de compras | Órdenes de compra emitidas |\n"
       "| Monto total (log) | ln(1 + suma del monto neto en CLP) |\n"
       "| N.º de proveedores | Proveedores distintos por RUT |\n"
       "| HHI | Σ (participación % de cada proveedor)²; máximo 10.000 |\n"
       "| CR1 | Participación del principal proveedor en el monto |\n"
       "| % trato directo | Fracción de OC por trato directo |\n"
       "| Monto medio (log) | ln(1 + monto medio por OC) |\n"
       "| % proveedores nuevos | Proveedores cuya primera OC con el organismo en el año cae en el 2º semestre |\n"
       "| N.º de rubros | Rubros (nivel 1) distintos comprados |\n"
       "| % con un solo oferente | Licitaciones adjudicadas con un único oferente |"),
    code("from observacompras import perfil\nperfiles = perfil.ejecutar()\n"
         "display(perfiles.groupby('anio').size())\n"
         "display(perfiles[C.VARIABLES].describe().T.round(3))"),
    code("import matplotlib.pyplot as plt\n"
         "fig, axs = plt.subplots(2, 5, figsize=(16, 5))\n"
         "for ax, v in zip(axs.ravel(), C.VARIABLES):\n"
         "    for a in C.ANIOS:\n"
         "        ax.hist(perfiles.loc[perfiles.anio == a, v], bins=30, alpha=.6, label=str(a))\n"
         "    ax.set_title(C.NOMBRES_VARIABLES[v], fontsize=9)\n"
         "axs[0, 0].legend(); plt.tight_layout(); plt.show()"),
])

guardar("02_autoencoder.ipynb", [
    md("# 02 · Detector de patrones atípicos: autoencoder (F3, H1)\n\n" + ENCABEZADO +
       "Aprendizaje **no supervisado** (Sesión 2): el autoencoder aprende a reconstruir el perfil de compra "
       "típico; el error de reconstrucción mide la atipicidad y su desglose por variable es la explicación.\n\n"
       "* Arquitectura: Dense 16 → 4 (latente) → 16 → 10, ReLU en capas ocultas y sigmoide en la salida.\n"
       "* Pérdida `mse` (regresión a valores en [0, 1]); optimizador RMSprop.\n"
       "* Validación K-fold con K = 5 (pocos cientos de organismos), early stopping (patience = 20).\n"
       "* Búsqueda: learning rate ∈ {1e-2, 1e-3, 1e-4} × batch size ∈ {8, 16, 32}; hasta 500 epochs.\n"
       "* Modelo final entrenado con todo 2023 por el número de epochs óptimo (como en Boston Housing).\n\n"
       "> **Tiempo estimado:** la búsqueda completa son 45 entrenamientos (9 combinaciones × 5 folds) más 6 "
       "finales; en CPU toma entre 40 y 60 minutos. La red es pequeña, por lo que la GPU no acelera mucho."),
    code(PREPARACION),
    code("import inspect\nfrom observacompras import autoencoder as AE\n"
         "print(inspect.getsource(AE.construir_autoencoder))\nAE.construir_autoencoder().summary()"),
    md("## Entrenamiento, puntajes y evaluación de H1\n\n"
       "La evaluación inyecta 20 organismos artificiales (5 tipos × 4) y mide qué fracción queda en el 5% "
       "superior del puntaje, frente a la línea base de puntaje z por variable."),
    code("from observacompras import pipeline_ae\nimport json\n"
         "resultado = pipeline_ae.ejecutar(verbose=0)\n"
         "print(json.dumps(resultado, indent=2, ensure_ascii=False))"),
    md("## Curvas de pérdida y búsqueda de hiperparámetros"),
    code("from IPython.display import Image, display\n"
         "for f in ['ae_curva_perdida.png', 'ae_busqueda.png', 'ae_recall_por_tipo.png']:\n"
         "    display(Image(str(C.DIR_FIGURAS / f)))"),
    code("import pandas as pd\n"
         "tabla = pd.read_csv(C.DIR_REPORTES / 'ae_busqueda_hiperparametros.csv')\n"
         "display(tabla.groupby(['learning_rate', 'batch_size'])[['mejor_val_loss', 'mejor_epoch']].mean().round(5))"),
    md("## Ranking de organismos atípicos del año de evaluación"),
    code("punt = pd.read_parquet(C.DIR_PROCESADOS / 'puntajes_atipicidad.parquet')\n"
         "top = punt[punt.anio == C.ANIO_PRUEBA].nsmallest(10, 'ranking')\n"
         "pd.set_option('display.max_colwidth', 200)\n"
         "display(top[['ranking', 'organismo_nombre', 'puntaje', 'percentil', 'explicacion']])"),
    code("display(pd.read_csv(C.DIR_REPORTES / 'ae_anomalias_sinteticas.csv'))"),
])

guardar("03_clasificador_unspsc.ipynb", [
    md("# 03 · Clasificador UNSPSC con transfer learning y búsqueda semántica (F4, F5, H2)\n\n" + ENCABEZADO +
       "Aprendizaje **supervisado** multiclase de etiqueta única. Como entrenar una red de lenguaje desde cero "
       "exigiría millones de textos, se usa **Sentence-BERT multilingüe** "
       "(`paraphrase-multilingual-MiniLM-L12-v2`) como base congelada que extrae un vector de 384 dimensiones "
       "por licitación, de forma análoga a una base convolucional preentrenada. Sobre esos vectores se entrena "
       "Dense 128 (ReLU) → Dropout 0,3 → Dense n_clases (softmax), con Adam y `categorical_crossentropy`.\n\n"
       "* Etiqueta: familia UNSPSC (4 dígitos) más frecuente entre los ítems de la licitación; se descartan "
       "familias con menos de 30 ejemplos.\n"
       "* Validación 85/15 sobre 2023; prueba final con 2024 (una sola vez).\n"
       "* Búsqueda: learning rate ∈ {1e-2, 1e-3, 1e-4} × batch size ∈ {64, 128, 256}; hasta 100 epochs.\n"
       "* Línea base: TF-IDF + regresión logística.\n\n"
       "> Se recomienda activar GPU en Colab (Entorno de ejecución → Cambiar tipo de entorno) para calcular "
       "los embeddings."),
    code(PREPARACION),
    code("import inspect\nfrom observacompras import clasificador as K\n"
         "print(inspect.getsource(K.construir_clasificador))\nK.construir_clasificador(60).summary()"),
    code("from observacompras import pipeline_clf\nimport json\n"
         "resultado = pipeline_clf.ejecutar(verbose=0)\n"
         "print(json.dumps(resultado, indent=2, ensure_ascii=False))"),
    md("## Curvas de pérdida y comparación con la línea base"),
    code("from IPython.display import Image, display\n"
         "for f in ['clf_curva_perdida.png', 'clf_busqueda.png', 'clf_comparacion.png']:\n"
         "    display(Image(str(C.DIR_FIGURAS / f)))"),
    md("## F5 · Búsqueda semántica\n\nDos licitaciones son similares si sus vectores tienen alta similitud coseno."),
    code("import numpy as np, pandas as pd\nfrom observacompras.busqueda import BuscadorSemantico\n"
         "from observacompras import embeddings as EMB\n"
         "lic = pd.read_parquet(C.DIR_PROCESADOS / 'licitaciones_app.parquet')\n"
         "E = np.load(C.DIR_PROCESADOS / 'embeddings_licitaciones.npy')\n"
         "b = BuscadorSemantico(E, lic)\n"
         "q = EMB.codificar(['mantención de equipos de rayos X'])[0]\n"
         "display(b.por_vector(q, 10)[['similitud', 'NroLicitacion', 'Institucion', 'NombreLicitacion', 'familia']])"),
    md("## Preparación de la vista de detalle de la aplicación\n\n"
       "Precalcula la sugerencia del clasificador para las órdenes de compra de los 30 organismos más atípicos."),
    code("from observacompras import pipeline_app\npipeline_app.ejecutar()"),
])

guardar("04_evaluacion.ipynb", [
    md("# 04 · Reporte de evaluación (Tabla 4)\n\n" + ENCABEZADO +
       "Consolida las seis métricas con sus líneas base y metas. H3 (explicabilidad) y H4 (utilidad) "
       "requieren juicio humano: complete `reports/revision_explicabilidad_H3.csv` y "
       "`reports/pruebas_usuario_H4.csv` (ver `docs/guia_pruebas_usuario.md`) y vuelva a ejecutar este cuaderno."),
    code(PREPARACION),
    code("from observacompras import reporte\nfrom IPython.display import Markdown\n"
         "Markdown(reporte.generar())"),
    md("## Subir los resultados al repositorio\n\n"
       "Los datos procesados, modelos y reportes quedan en la carpeta del repositorio en Drive. Para "
       "publicarlos en GitHub, desde una terminal local o desde Colab con un token personal:\n\n"
       "```bash\ngit add data/processed models reports\ngit commit -m \"Resultados del pipeline 2023-2024\"\ngit push\n```"),
])
print("Cuadernos generados en notebooks/")
