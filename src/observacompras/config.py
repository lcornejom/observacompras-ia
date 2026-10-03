"""Configuración central de ObservaCompras IA.

Todas las decisiones de diseño del documento (Tablas 2, 3 y 4) y las
decisiones de implementación que el documento deja abiertas están aquí,
con referencia a la sección que las origina. Ver también docs/decisiones.md.
"""
from pathlib import Path

# --------------------------------------------------------------------------
# Rutas
# --------------------------------------------------------------------------
RAIZ = Path(__file__).resolve().parents[2]
DIR_RAW = RAIZ / "data" / "raw"
DIR_PROCESADOS = RAIZ / "data" / "processed"
DIR_MODELOS = RAIZ / "models"
DIR_REPORTES = RAIZ / "reports"
DIR_FIGURAS = DIR_REPORTES / "figuras"

# --------------------------------------------------------------------------
# Reproducibilidad (Sección 9: "las semillas aleatorias se publicarán")
# --------------------------------------------------------------------------
SEMILLA = 42

# --------------------------------------------------------------------------
# F1 - Ingesta (Sección 4)
# --------------------------------------------------------------------------
SECTOR = "SALUD"
ANIO_ENTRENAMIENTO = 2023   # Tabla 3: 85/15 sobre 2023
ANIO_PRUEBA = 2024          # Tabla 3: prueba final con 2024
ANIOS = (ANIO_ENTRENAMIENTO, ANIO_PRUEBA)

# Estados de OC que no corresponden a compras efectivas
ESTADOS_OC_EXCLUIDOS = {"Solicitud de Cancelacion", "Cancelada", "Eliminada", "Guardada"}
# "licitaciones ... adjudicadas" (Sección 4)
ESTADO_LICITACION_VALIDO = "Adjudicada"

# Decisión (f): RUT numérico menor a 50.000.000 => persona natural.
# Se anonimiza conforme a Ley 19.628 y Ley 21.719 (Tabla 5).
UMBRAL_RUT_PERSONA_JURIDICA = 50_000_000
SAL_ANONIMIZACION = "observacompras-ia-mti493"

# --------------------------------------------------------------------------
# F2 - Perfil de compra: 10 variables por organismo y año (Tabla 2)
# --------------------------------------------------------------------------
VARIABLES = [
    "n_compras",
    "monto_total_log",
    "n_proveedores",
    "hhi",
    "cr1",
    "pct_trato_directo",
    "monto_medio_log",
    "pct_proveedores_nuevos",
    "n_rubros",
    "pct_oferente_unico",
]
NOMBRES_VARIABLES = {
    "n_compras": "N.º de compras",
    "monto_total_log": "Monto total (log)",
    "n_proveedores": "N.º de proveedores",
    "hhi": "Índice HHI",
    "cr1": "Participación del principal proveedor (CR1)",
    "pct_trato_directo": "% de trato directo",
    "monto_medio_log": "Monto medio (log)",
    "pct_proveedores_nuevos": "% de proveedores nuevos",
    "n_rubros": "N.º de rubros",
    "pct_oferente_unico": "% de compras con un solo oferente",
}
# Frases para la explicación (lenguaje descriptivo, nunca califica conductas; Tabla 5)
FRASES_VARIABLES = {
    "n_compras": "su número de compras",
    "monto_total_log": "su monto total adjudicado",
    "n_proveedores": "su número de proveedores",
    "hhi": "la concentración del gasto entre proveedores (HHI)",
    "cr1": "la concentración en su principal proveedor",
    "pct_trato_directo": "su porcentaje de trato directo",
    "monto_medio_log": "el monto medio de sus compras",
    "pct_proveedores_nuevos": "su porcentaje de proveedores nuevos",
    "n_rubros": "la cantidad de rubros que compra",
    "pct_oferente_unico": "su porcentaje de licitaciones con un solo oferente",
}
# Mínimo de órdenes de compra para incluir un organismo-año en el perfil
# (evita perfiles calculados sobre muy pocas compras).
MIN_OC_POR_ORGANISMO = 20

# --------------------------------------------------------------------------
# F3 - Autoencoder (Sección 5.1 y Tabla 3)
# --------------------------------------------------------------------------
AE_CAPAS = (16, 4, 16)            # Dense 16 -> 4 -> 16 -> 10
AE_EPOCHS = 500
AE_PACIENCIA = 20
AE_LEARNING_RATES = (1e-2, 1e-3, 1e-4)
AE_BATCH_SIZES = (8, 16, 32)
AE_K_FOLDS = 5

# Evaluación H1 (Tabla 4)
N_ANOMALIAS_SINTETICAS = 20       # 5 tipos x 4 casos (decisión d)
PERCENTIL_ALERTA = 95             # "5% superior del puntaje"
TOP_N_ESTABILIDAD = 10

# --------------------------------------------------------------------------
# F4 - Clasificador UNSPSC (Sección 5.2 y Tabla 3)
# --------------------------------------------------------------------------
MODELO_SBERT = "paraphrase-multilingual-MiniLM-L12-v2"
DIM_EMBEDDING = 384
NIVEL_UNSPSC = 4                  # decisión (c): familia = 4 dígitos
MIN_EJEMPLOS_CLASE = 30           # Tabla 5
CLF_DENSA = 128
CLF_DROPOUT = 0.3
CLF_EPOCHS = 100
CLF_PACIENCIA = 20               # el documento reutiliza el mismo callback `parada`
CLF_LEARNING_RATES = (1e-2, 1e-3, 1e-4)
CLF_BATCH_SIZES = (64, 128, 256)
CLF_VALIDACION = 0.15
TOP_K_SUGERENCIAS = 3

# --------------------------------------------------------------------------
# F6 - Aplicación (Sección 6)
# --------------------------------------------------------------------------
ADVERTENCIA = (
    "Un patrón atípico no es evidencia de irregularidad; "
    "indica dónde conviene mirar."
)
