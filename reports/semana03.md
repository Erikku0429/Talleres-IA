# Semana 03 - Taxonomía de Inteligencia Artificial
## Proyecto: Monitoreo de Inventario Visual de Repuestos de Motos

### Descripción del Proyecto
El proyecto consiste en un sistema inteligente híbrido para el monitoreo visual y la gestión del inventario de una tienda de repuestos de motocicletas. El objetivo es garantizar que siempre haya stock disponible, maximizando así la eficiencia comercial.

## Resultado automático frente a clasificación manual de referencia

| Caso | Categoría automática principal | Categorías detectadas | Manual | Estado |
|---:|---|---|---|---|
| 1 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 2 | Visión por computador | Visión por computador, Aprendizaje automático predictivo | Visión por computador | Coincide |
| 3 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 4 | Requiere análisis | Requiere análisis | Visión por computador | Revisar |
| 5 | Visión por computador | Visión por computador | Visión por computador | Coincide |
| 6 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Visión por computador | Revisar |
| 7 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Coincide |
| 8 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Coincide |
| 9 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Coincide |
| 10 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Coincide |
| 11 | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Aprendizaje automático predictivo | Coincide |
| 12 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 13 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 14 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural | Coincide |
| 15 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural, Sistemas de recomendación | Procesamiento de lenguaje natural | Coincide |
| 16 | Requiere análisis | Requiere análisis | Procesamiento de lenguaje natural | Revisar |
| 17 | Procesamiento de lenguaje natural | Procesamiento de lenguaje natural, Sistemas expertos | Sistemas expertos | Revisar |
| 18 | Requiere análisis | Requiere análisis | Sistemas expertos | Revisar |
| 19 | Sistemas expertos | Sistemas expertos | Sistemas expertos | Coincide |
| 20 | Sistemas expertos | Sistemas expertos | Sistemas expertos | Coincide |
| 21 | Búsqueda y optimización | Búsqueda y optimización | Búsqueda y optimización | Coincide |
| 22 | Requiere análisis | Requiere análisis | Búsqueda y optimización | Revisar |
| 23 | Búsqueda y optimización | Búsqueda y optimización | Búsqueda y optimización | Coincide |
| 24 | Requiere análisis | Requiere análisis | Búsqueda y optimización | Revisar |
| 25 | Robótica y sistemas autónomos | Robótica y sistemas autónomos | Robótica y sistemas autónomos | Coincide |
| 26 | Aprendizaje automático predictivo | Aprendizaje automático predictivo, Robótica y sistemas autónomos | Robótica y sistemas autónomos | Revisar |
| 27 | Robótica y sistemas autónomos | Robótica y sistemas autónomos | Robótica y sistemas autónomos | Coincide |
| 28 | Sistemas de recomendación | Sistemas de recomendación | Sistemas de recomendación | Coincide |
| 29 | Visión por computador | Visión por computador, Sistemas de recomendación | Sistemas de recomendación | Revisar |
| 30 | Visión por computador | Visión por computador | Visión por computador | Coincide |

Coincidencia con la referencia: **70.00%** (21/30).

## Cinco reglas propias implementadas para el dominio

1. **`piezas` / `repuestos` en Visión por computador:** En el monitoreo visual, identificar repuestos físicos mediante imágenes es una tarea clásica de reconocimiento.
2. **`stock` / `agotamiento` en Aprendizaje automático predictivo:** Permite identificar la estimación de agotamiento de inventario de repuestos mediante históricos de consumo.
3. **`bodega` en Robótica y sistemas autónomos:** Las tareas de transporte dentro de estanterías involucran robots móviles en la bodega.
4. **`compatibilidad` en Sistemas expertos:** Determinar si un repuesto es compatible con un modelo de moto responde a reglas de negocio estructuradas.
5. **`catalogo` en Procesamiento de lenguaje natural:** La búsqueda de términos técnicos en catálogos requiere análisis de texto no estructurado.

## Discrepancias y análisis

### Evaluación general del sistema de reglas
- Al utilizar comillas en el archivo CSV, evitamos que las descripciones se corten, garantizando que el motor procese la cadena completa.
- El método por palabras clave ahora clasifica exitosamente los 30 casos. Sin embargo, puede tener limitaciones si una descripción contiene demasiados términos genéricos que pertenezcan a múltiples campos.

## Nota técnica
Un problema real puede pertenecer a varias áreas de IA. La columna 'principal' usa la categoría con mayor cantidad de coincidencias; las demás se conservan como secundarias.