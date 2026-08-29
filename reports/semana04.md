Semana 04 - Marco Tecnológico de la IA: Búsqueda A* y Algoritmos Adversariales
Proyecto: Sistema de Monitoreo de Inventario para Tienda de Motos

A. Descripción del Problema
En la bodega de repuestos necesitamos que el operario encuentre la forma más rápida de ir por las piezas pedidas (como kits de arrastre, aceite o pastillas de freno).El objetivo es armar la ruta más corta evitando estantes bloqueados y pasillos llenos de gente.

B. Definición del Problema en Estados
 --Estado Inicial: Coordenada filas,columnas donde arranca el operario.
 --Posibles Estados: Cualquier casilla caminable filas,columnas dentro de la bodega.
 --Acciones: Moverse un paso hacia arriba,abajo,izquierda,derecha.
 --Transiciones: Pasar a la casilla de al lado si no hay un muro o estante estorbando.
 --Meta: Coordenada fila_meta,columna_meta de la estantería del repuesto.
 --Costo de Camino g(n): Se validan los costos de cada accion, pasillo libre 1.0 y zona concurrida 3.0
 --Heurística h(n): Distancia Euclidiana (línea recta) desde la casilla actual hasta la meta.
 --Criterio de Selección: Escoger la ruta que dé el menor resultado en $f(n) = g(n) + h(n)$.

C. Pruebas de Ejecución
 --Prueba 1: Ir directo desde recepción hasta el estante de cascos
 --Entrada: Desde `(0,0)` hasta `(2,4)`.
 --Ruta Calculada: `(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,3) -> (2,4)`
 --Costo final $g(n)$: `6.0` | Casillas revisadas: `8`
 --Explicación**: El algoritmo evalúa el mapa y selecciona el camino que evita estantes y zonas de tráfico, equilibrando la distancia recorrida con lo que falta para llegar.

 --Prueba 2: Ruta evitando el paso congestionado del taller de mecánicos
 --Entrada: Desde `(0,0)` hasta `(5,3)`.
 --Ruta Calculada: `(0,0) -> (0,1) -> (0,2) -> (1,2) -> (2,2) -> (2,3) -> (2,4) -> (3,4) -> (4,4) -> (4,3) -> (5,3)`
 --Costo final $g(n)$: `10.0` | Casillas revisadas: `13`
 --Explicación**: El algoritmo evalúa el mapa y selecciona el camino que evita estantes y zonas de tráfico, equilibrando la distancia recorrida con lo que falta para llegar.

 --Prueba 3: Recorrido largo desde el fondo de la bodega a la zona de despacho
 --Entrada: Desde `(5,0)` hasta `(0,5)`.
 --Ruta Calculada: `(5,0) -> (4,0) -> (3,0) -> (2,0) -> (2,1) -> (2,2) -> (2,3) -> (2,4) -> (2,5) -> (1,5) -> (0,5)`
 --Costo final $g(n)$: `10.0` | Casillas revisadas: `21`
 --Explicación**: El algoritmo evalúa el mapa y selecciona el camino que evita estantes y zonas de tráfico, equilibrando la distancia recorrida con lo que falta para llegar.

D. Resultados
--¿Aplica al proyecto?: No aplica, porque los estantes no hacen daño al negocio.
--Diferencia técnica: La búsqueda $A^*$ sirve para encontrar una ruta eficiente. Minimax se usa cuando hay un rival pensante tratando de contrarrestar nuestras decisiones.
--Poda Alfa-Beta: Este ayuda a quitar opciones que sabemos que no se van a tomar
--Prueba realizada: En un árbol de ejemplo de 3 niveles, el algoritmo seleccionó correctamente la mejor decisión con valor de `12`.

E. Análisis
--Lo bueno: $A^*$ siempre encuentra la ruta más rápida si la estimación de distancia no exagera el costo real.
--Por mejorar: hasta el momento todo funciona con datos establecidos