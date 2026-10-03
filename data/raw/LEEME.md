# Datos crudos

Copie aquí los archivos del sector Salud descargados de https://datos-abiertos.chilecompra.cl
(órdenes de compra y licitaciones de 2023 y 2024), en `.7z`, `.zip` o `.csv`.

La ingesta reconoce cada archivo por su encabezado (`codigoOC;...` para órdenes de compra y
`NroLicitacion;NombreLicitacion;...` para licitaciones), así que los nombres de archivo no importan.
Esta carpeta no se versiona en Git por el tamaño de los archivos (~3,5 GB por semestre).

**Importante:** deje los archivos comprimidos tal como se descargan (un `.7z` o `.zip` por semestre).
Los CSV de distintos semestres tienen el mismo nombre (por ejemplo, `07OCCompraAgil.csv`), así que si
los descomprime a mano en esta misma carpeta se sobrescriben. La ingesta descomprime cada archivo en
una carpeta temporal separada.
