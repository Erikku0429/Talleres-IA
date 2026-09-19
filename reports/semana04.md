# Semana 04 - Marco Tecnológico de la IA: Búsqueda A* y Algoritmos Adversariales
## Proyecto: Monitoreo de Inventario Visual para Tienda de Motos

### A. Descripción del Problema
En la gestión de una tienda de repuestos para motocicletas, el tiempo de respuesta del operario al buscar piezas solicitadas (como kits de arrastre, bujías o pastillas de freno) is crítico para mantener la operatividad comercial. El sistema utiliza búsqueda heurística para determinar la ruta de recolección (*picking*) más corta dentro de la bodega, evitando estanterías bloqueadas y penalizando zonas de congestión (como el área del taller).

### B. Definición del Problema en Espacio de Estados
- **Estado Inicial:** Coordenada $(fila, columna)$ donde se ubica el operario al recibir la orden.
- **Espacio de Estados:** Todas las casillas transitables $(fila, columna)$ dentro de la cuadrícula de la bodega.
- **Acciones / Operadores:** Desplazamiento ortogonal (arriba, abajo, izquierda, derecha).
- **Transición:** Cambio de posición a una celda vecina si no constituye un obstáculo físico (estantería).
- **Meta:** Coordenada $(fila, columna)$ exacta del estante donde se almacena el repuesto requerido.
- **Costo de Camino $g(n)$:** Variable según el terreno (1.0 para pasillos libres, 3.0 para zonas de alta congestión).
- **Heurística $h(n)$:** Distancia Manhattan, óptima para espacios en cuadrícula sin movimientos diagonales.
- **Función de Prioridad:** $f(n) = g(n) + h(n)$.

### C. Pruebas de Ejecución (Algoritmo A*)
#### Prueba 1: Ruta directa desde recepción hasta el estante principal de cascos y visores
- **Trayecto:** Desde `(0,0)` hasta `(2,4)`.
- **Ruta Calculada:** `(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,3) -> (2,4)`
- **Costo Total $g(n)$:** `6.0` | **Nodos Evaluados:** `8`
- **Justificación:** El algoritmo equilibra la distancia real y los costos de congestión del terreno para trazar el camino más eficiente sin colisionar con el inventario físico.

#### Prueba 2: Ruta de abastecimiento evitando el pasillo congestionado del taller de servicio
- **Trayecto:** Desde `(0,0)` hasta `(5,3)`.
- **Ruta Calculada:** `(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,3) -> (2,4) -> (3,4) -> (4,4) -> (4,3) -> (5,3)`
- **Costo Total $g(n)$:** `10.0` | **Nodos Evaluados:** `13`
- **Justificación:** El algoritmo equilibra la distancia real y los costos de congestión del terreno para trazar el camino más eficiente sin colisionar con el inventario físico.

#### Prueba 3: Recorrido logístico desde el fondo de la bodega hasta la zona de despacho de repuestos
- **Trayecto:** Desde `(5,0)` hasta `(0,5)`.
- **Ruta Calculada:** `(5,0) -> (4,0) -> (3,0) -> (2,0) -> (2,1) -> (2,2) -> (2,3) -> (2,4) -> (2,5) -> (1,5) -> (0,5)`
- **Costo Total $g(n)$:** `10.0` | **Nodos Evaluados:** `14`
- **Justificación:** El algoritmo equilibra la distancia real y los costos de congestión del terreno para trazar el camino más eficiente sin colisionar con el inventario físico.

### D. Análisis de Decisiones Adversariales (Minimax y Poda Alfa-Beta)
- **Aplicabilidad al Proyecto:** Los estantes y componentes de la tienda son pasivos y no actúan como adversarios racionales. Sin embargo, el marco teórico de Minimax se modela para escenarios de negociación automatizada con proveedores de repuestos o asignación competitiva de recursos.
- **Resultado de Simulación:** En un árbol de decisión de profundidad 3, el algoritmo evaluó el escenario óptimo devolviendo un valor de utilidad de `12`.
- **Eficiencia (Poda Alfa-Beta):** Permitió descartar ramas de decisión redundantes que no afectaban el resultado final, reduciendo el tiempo de cómputo.

### E. Conclusiones y Trade-offs
- **Optimalidad:** $A^*$ garantiza encontrar la ruta de menor costo en la bodega gracias a la admisibilidad de la heurística de Manhattan.
- **Rendimiento:** El uso de colas de prioridad (*heapq*) optimiza la selección del nodo prometedor en tiempo logarítmico.