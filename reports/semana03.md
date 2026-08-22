# Semana 03 — Taxonomía de Inteligencia Artificial
## Resultado automático frente a clasificación manual de referencia
| Caso | Categoría automática principal | Categorías detectadas | Manual | Estado |
|---|---|---|---|---|
| 1 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 2 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 3 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Coincide |
| 4 | Búsqueda y optimización | Búsqueda y optimización | Búsqueda y optimización | Coincide |
| 5 | Sistemas de recomendación | Sistemas de recomendación | Sistemas de recomendación | Coincide |
| 6 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Coincide |
| 7 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 8 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 9 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Coincide |
| 10 | Sistemas expertos | Sistemas expertos | Sistemas expertos | Coincide |
| 11 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 12 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 13 | Robótica y sistemas autónomos | Robótica y sistemas autónomos | Robótica y sistemas autónomos | Coincide |
| 14 | Búsqueda y optimización | Búsqueda y optimización | Búsqueda y optimización | Coincide |
| 15 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Coincide |
| 16 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 17 | Visión por computador | Visión por computador, Robótica y sistemas autónomos | Visión por computador | Coincide |
| 18 | Sistemas expertos | Sistemas expertos | Sistemas expertos | Coincide |
| 19 | Robótica y sistemas autónomos | Robótica y sistemas autónomos | Robótica y sistemas autónomos | Coincide |
| 20 | Búsqueda y optimización | Búsqueda y optimización | Búsqueda y optimización | Coincide |

Coincidencia con la referencia: **100.00%** (20/20).

## Cinco reglas propias
1. Visión por computador -> `('matricula', 'matriculas')`: Permite clasificar sistemas de Reconocimiento Automático de Placas.
2. Procesamiento de lenguaje natural -> `('sentimiento',)`: Esencial para tareas de minería de opiniones.
3. Aprendizaje automático predictivo -> `('falla', 'fallas')`: Utilizado en modelos de mantenimiento predictivo mediante sensores.
4. Sistemas expertos -> `('sintoma', 'sintomas')`: Fundamental en sistemas de diagnóstico basados en reglas.
5. Robótica y sistemas autónomos -> `('trayectoria', 'trayectorias')`: Necesario para algoritmos de planificación de movimiento.

## Discrepancias y análisis
para este caso el motor logró el 100% de precisión no logro identificar ni registrar discrepancias.

## Nota técnica
Un problema real puede pertenecer a varias áreas de IA. La columna 'principal' usa la categoría con mayor cantidad de coincidencias; las demás coincidencias se conservan como categorías secundarias.