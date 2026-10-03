# Decisiones de implementación

El código sigue el documento *ObservaCompras IA — Propuesta de MVP (v4)*. Esta página registra las
decisiones que el documento no especifica, con su justificación. Todas son parámetros de
`src/observacompras/config.py` y pueden cambiarse sin tocar el resto del código.

## Fieles al documento (sin cambios)

| Elemento | Documento | Implementación |
|---|---|---|
| Datos | Licitaciones y OC adjudicadas, Salud, 2023–2024 | `ingesta.py`, filtro `Sector == SALUD` |
| 10 variables por organismo y año | Tabla 2 (F2) | `perfil.py` |
| Autoencoder | Dense 16 → 4 → 16 → 10, ReLU/sigmoide, mse, RMSprop | `autoencoder.construir_autoencoder` |
| Puntaje y explicación | MSE de reconstrucción; error por variable | `autoencoder.puntuar` |
| Clasificador | SBERT `paraphrase-multilingual-MiniLM-L12-v2` congelado + Dense 128 → Dropout 0,3 → softmax, Adam | `clasificador.construir_clasificador` |
| Hiperparámetros | lr {1e-2, 1e-3, 1e-4}; batch {8, 16, 32} y {64, 128, 256}; 500 y 100 epochs | `config.py` |
| Early stopping | `val_loss`, patience 20, `restore_best_weights` | Ambos modelos (el documento reutiliza el mismo callback) |
| Validación | K-fold K = 5 (autoencoder); 85/15 sobre 2023 y prueba 2024 (clasificador) | `pipeline_ae.py`, `pipeline_clf.py` |
| Líneas base | Puntaje z por variable; TF-IDF + regresión logística | `anomalias.LineaBaseZ`, `clasificador.LineaBaseTfidf` |
| Búsqueda semántica | Similitud coseno sobre los mismos 384-dim | `busqueda.py` |
| App | Streamlit, 3 vistas, advertencia fija, periodo y tipos de compra | `app/streamlit_app.py` |
| Privacidad | Personas naturales anonimizadas | `ingesta.anonimizar_proveedores` |
| Plazo (Tabla 5) | F5 se reduce a palabras clave si hay problemas | Respaldo automático `BuscadorPalabrasClave` |

## Decisiones que el documento deja abiertas

**a. % de proveedores nuevos.** Compararlo con el año anterior exige datos de 2022, que quedan fuera
del alcance. Además, la variable debe definirse igual en 2023 (entrenamiento) y 2024 (evaluación)
para no introducir un cambio artificial de distribución. Por eso se define dentro de cada año: la
fracción de proveedores cuya primera OC con el organismo en ese año cae en el segundo semestre.

**b. Años del autoencoder.** Se entrena con los organismos de 2023 y se puntúa 2024, igual que el
clasificador. El ranking que muestra la aplicación por defecto es el de 2024 (Sección 6).

**c. Nivel UNSPSC.** Familia (4 dígitos). La etiqueta de una licitación es la familia más frecuente
entre sus ítems distintos (desempate por el código menor). Con el primer semestre de 2023 quedan 65
familias con al menos 30 ejemplos, que cubren el 87% de las licitaciones.

**d. Anomalías sintéticas.** 20 = 5 tipos × 4 casos. Cada caso parte de un organismo real de 2024
elegido al azar y modifica solo las variables del patrón:
`proveedor_unico` (CR1 0,93–0,98 y su HHI), `trato_directo` (85–100%), `oferente_unico` (85–100%),
`monto_extremo` (monto medio 10 veces el máximo de 2023) y `combinacion` (concentración, trato
directo, oferente único y proveedores nuevos en su percentil 90, con pocos proveedores y rubros:
cada variable es normal por separado, que es el caso que motiva el Problema 1 del documento).

**Recall@5%.** Cada anomalía se compara por separado con los organismos reales de 2024: se cuenta
como detectada si su puntaje supera el percentil 95 de los reales. Inyectar las 20 a la vez haría
imposible la meta, porque el 5% de unos 240 organismos son solo 12 lugares.

**e. Línea base z.** Puntaje = máximo |z| entre las 10 variables, con media y desviación de 2023.

**f. Persona natural.** RUT numérico menor a 50.000.000. Se reemplazan RUT y nombre por un seudónimo
estable (`PN-` + hash con sal), y el nombre de la OC se oculta porque suele contener el nombre de la
persona. Al ser estable, el seudónimo no altera HHI, CR1 ni proveedores nuevos.

**g. Embeddings para la app.** Se precalculan en Colab y se guardan en el repositorio, para que la
app funcione sin GPU. El modelo de lenguaje solo se carga si el usuario escribe un texto nuevo.

## Otras precisiones

* **Organismo** = institución (`entCode`). Con el primer semestre de 2023 son 224, lo que coincide con
  los "pocos cientos de organismos" del documento. Se exige un mínimo de 20 OC por organismo y año.
* **Año de una licitación** = año de `FechaCierre`, porque los archivos mensuales del portal se agrupan
  por fecha de cierre.
* **Estados de OC excluidos:** solicitudes de cancelación, canceladas, eliminadas y guardadas.
* **% de trato directo** se calcula sobre el número de OC; **HHI y CR1**, sobre el monto neto en CLP.
* **Organismos sin licitaciones:** el % con un solo oferente se imputa con la mediana de 2023 y se
  marca en `oferente_unico_imputado`.
* **Modelo final del autoencoder:** como en el ejemplo de Boston Housing, el K-fold elige la
  combinación de hiperparámetros y el número de epochs (mediana del mejor epoch de cada fold, porque el
  early stopping detiene cada fold en un epoch distinto); luego
  se entrena con todos los organismos de 2023. La estabilidad del ranking se mide con los 5 modelos
  del K-fold (coincidencia media del top 10 entre pares de folds).
* **Clasificador:** como Keras toma el 15% final como validación, los datos se barajan con la semilla
  antes. Se conserva el modelo de la combinación con menor `val_loss`. La línea base usa la misma
  partición y elige su regularización (C) con la misma validación.
* **Escalamiento:** min–max con los valores de 2023; en 2024 los valores fuera de rango se recortan a
  [0, 1], porque la salida sigmoide no puede superar ese intervalo.
* **Semilla:** 42 en NumPy, Keras/TensorFlow y scikit-learn.

## Modo demostración

`--modo demo` permite probar todo el pipeline con un solo semestre: el 1er trimestre cumple el rol de
2023 y el 2º trimestre, el de 2024. Sus resultados validan el código, no las hipótesis.
