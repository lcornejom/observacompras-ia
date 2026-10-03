# Guía de pruebas con usuarios (H4 — Utilidad)

**Meta (Tabla 4):** mediana del tiempo de tarea menor a 5 minutos y SUS ≥ 70, con cinco usuarios no
técnicos. **Línea base:** la misma tarea con una planilla de órdenes de compra, sin la aplicación.

## Participantes

Cinco personas que sepan interpretar un indicador pero que no programen ni entrenen modelos (perfil
de la Sección 3). Registre solo un código de participante (P1 a P5), sin datos personales.

## Preparación

1. Abra la aplicación en el computador de prueba (`streamlit run app/streamlit_app.py`).
2. Explique en un minuto el objetivo: "La aplicación ayuda a decidir qué organismos revisar primero y
   por qué. Un patrón atípico no es evidencia de irregularidad."
3. No explique cómo usarla. Tome el tiempo de cada tarea desde que termina de leerla en voz alta.

## Tareas

| # | Tarea | Se considera resuelta cuando… |
|---|---|---|
| T1 | "Indique qué organismo conviene revisar primero en 2024 y por qué motivo." | Nombra el primer organismo del ranking y su variable o motivo principal. |
| T2 | "Para ese organismo, encuentre una compra a su principal proveedor cuya categoría podría estar mal registrada." | Señala una OC marcada "No — revisar codificación". |
| T3 | "Encuentre compras similares de otros hospitales y diga qué proveedor las adjudicó." | Muestra resultados del buscador y lee un proveedor adjudicado. |

La línea base repite T1 con una planilla Excel de las órdenes de compra del año, sin la aplicación
(`pd.read_parquet("data/processed/ordenes_compra_2024.parquet").to_excel("planilla.xlsx")`).

## Registro

Complete `reports/pruebas_usuario_H4.csv`, una fila por participante y tarea:

```
participante,tarea,con_app,tiempo_min,resuelta,sus,comentario
P1,T1,si,2.5,si,77.5,
```

La columna `sus` lleva el puntaje SUS del participante (el mismo valor en sus filas). Para H4 se usa
la mediana de `tiempo_min` de las filas con `con_app = si`.

## Cuestionario SUS (System Usability Scale)

Escala de 1 (muy en desacuerdo) a 5 (muy de acuerdo):

1. Creo que me gustaría usar este sistema con frecuencia.
2. Encontré el sistema innecesariamente complejo.
3. Pensé que el sistema era fácil de usar.
4. Creo que necesitaría el apoyo de una persona técnica para poder usar este sistema.
5. Encontré que las diversas funciones del sistema estaban bien integradas.
6. Pensé que había demasiada inconsistencia en este sistema.
7. Imagino que la mayoría de las personas aprendería a usar este sistema muy rápidamente.
8. Encontré el sistema muy engorroso de usar.
9. Me sentí muy seguro(a) usando el sistema.
10. Necesité aprender muchas cosas antes de poder empezar a usar este sistema.

**Cálculo:** para los ítems impares, reste 1 a la respuesta; para los pares, reste la respuesta a 5.
Sume los diez valores y multiplique por 2,5 (resultado entre 0 y 100).

## H3 — Explicabilidad

Los autores completan `reports/revision_explicabilidad_H3.csv`: para cada uno de los 10 organismos
más atípicos, escriben en `frase_del_revisor` el patrón en una frase y marcan
`descriptible_en_una_frase` con "si" o "no". Meta: al menos 7 de 10.
