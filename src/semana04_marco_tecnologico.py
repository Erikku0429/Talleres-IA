import math
import heapq
from dataclasses import dataclass
from pathlib import Path

# Configuración de carpetas del proyecto
DIRECTORIO_RAIZ = Path(__file__).resolve().parent.parent
CARPETA_REPORTES = DIRECTORIO_RAIZ / "reports"
RUTA_REPORTE = CARPETA_REPORTES / "semana04.md"


# 1. BÚSQUEDA A*: RUTA ÓPTIMA EN BODEGA DE REPUESTOS

@dataclass(frozen=True)
class CasillaBodega:
    fila: int
    columna: int

    def __lt__(self, otra):
        return (self.fila, self.columna) < (otra.fila, otra.columna)


class MapaBodega:
    """
    Representa el mapa de la bodega de repuestos como una cuadrícula de casillas:
    0 = Pasillo despejado (Costo estándar: 1.0)
    1 = Estantería de repuestos / Obstáculo (Bloqueado)
    2 = Zona de alta congestión / Taller (Costo elevado por tráfico: 3.0)
    """

    def __init__(self, croquis: list[list[int]]):
        self.croquis = croquis
        self.filas = len(croquis)
        self.columnas = len(croquis[0])

    def es_paso_valido(self, casilla: CasillaBodega) -> bool:
        dentro_del_mapa = 0 <= casilla.fila < self.filas and 0 <= casilla.columna < self.columnas
        return dentro_del_mapa and self.croquis[casilla.fila][casilla.columna] != 1

    def obtener_costo_terreno(self, casilla: CasillaBodega) -> float:
        # Penalización de costo si el operario cruza por zonas de alta afluencia o tráfico del taller
        return 3.0 if self.croquis[casilla.fila][casilla.columna] == 2 else 1.0

    def obtener_casillas_vecinas(self, casilla: CasillaBodega) -> list[CasillaBodega]:
        # Movimientos ortogonales permitidos: arriba, abajo, izquierda, derecha
        pasos_posibles = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        vecinos = []
        for df, dc in pasos_posibles:
            siguiente = CasillaBodega(casilla.fila + df, casilla.columna + dc)
            if self.es_paso_valido(siguiente):
                vecinos.append(siguiente)
        return vecinos


def calcular_distancia_manhattan(origen: CasillaBodega, destino: CasillaBodega) -> float:
    """
    Heurística h(n) óptima y admisible para cuadrículas con movimientos ortogonales.
    Calcula la distancia Manhattan sin exagerar el costo real.
    """
    return float(abs(origen.fila - destino.fila) + abs(origen.columna - destino.columna))


def buscar_ruta_a_estrella(mapa: MapaBodega, punto_inicio: CasillaBodega, punto_meta: CasillaBodega):
    """
    Algoritmo A* para optimizar las rutas de picking (recolección de repuestos de motos).
    Minimiza f(n) = g(n) + h(n)
    """
    cola_prioridad = []
    heapq.heappush(cola_prioridad, (0.0, punto_inicio))

    rastreo_camino = {}
    costo_recorrido_g = {punto_inicio: 0.0}
    pasos_evaluados = 0

    while cola_prioridad:
        f_actual, casilla_actual = heapq.heappop(cola_prioridad)
        pasos_evaluados += 1

        if casilla_actual == punto_meta:
            ruta_final = []
            paso = casilla_actual
            while paso in rastreo_camino:
                ruta_final.append(paso)
                paso = rastreo_camino[paso]
            ruta_final.append(punto_inicio)
            ruta_final.reverse()
            return ruta_final, costo_recorrido_g[punto_meta], pasos_evaluados

        for vecina in mapa.obtener_casillas_vecinas(casilla_actual):
            costo_paso = mapa.obtener_costo_terreno(vecina)
            nuevo_costo_g = costo_recorrido_g[casilla_actual] + costo_paso

            if vecina not in costo_recorrido_g or nuevo_costo_g < costo_recorrido_g[vecina]:
                rastreo_camino[vecina] = casilla_actual
                costo_recorrido_g[vecina] = nuevo_costo_g
                estimacion_f = nuevo_costo_g + calcular_distancia_manhattan(vecina, punto_meta)
                heapq.heappush(cola_prioridad, (estimacion_f, vecina))

    return None, float('inf'), pasos_evaluados


# 2. PRÁCTICA DE REFERENCIA: MINIMAX CON PODA ALFA-BETA (ANÁLISIS ADVERSARIAL)

def probar_minimax_alfa_beta(nodo: int, nivel: int, turno_max: bool, valores: list[int], alfa: float,
                             beta: float) -> int:
    """
    Simulación de toma de decisiones adversariales.
    En el contexto de la tienda de repuestos, se incluye teóricamente para evaluar la optimización
    frente a un entorno competitivo o de negociación con proveedores.
    """
    if nivel == 3:
        return valores[nodo]

    if turno_max:
        mejor_opcion = -math.inf
        for i in range(2):
            puntaje = probar_minimax_alfa_beta(nodo * 2 + i, nivel + 1, False, valores, alfa, beta)
            mejor_opcion = max(mejor_opcion, puntaje)
            alfa = max(alfa, mejor_opcion)
            if beta <= alfa:
                break  # Poda alfa: elimina ramas innecesarias
        return mejor_opcion
    else:
        mejor_opcion = math.inf
        for i in range(2):
            puntaje = probar_minimax_alfa_beta(nodo * 2 + i, nivel + 1, True, valores, alfa, beta)
            mejor_opcion = min(mejor_opcion, puntaje)
            beta = min(beta, mejor_opcion)
            if beta <= alfa:
                break  # Poda beta: elimina ramas innecesarias
        return mejor_opcion


# 3. EJECUCIÓN DE PRUEBAS Y GENERACIÓN DE REPORTE EN MARKDOWN

def correr_pruebas_y_generar_informe():
    CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)

    # Plano de la bodega de repuestos de motos (6x6)
    # 0 = Pasillo, 1 = Estantería de repuestos (Obstáculo), 2 = Zona de alto tráfico / Taller
    plano_tienda = [
        [0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1],
        [0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ]
    mapa_bodega = MapaBodega(plano_tienda)

    casos_de_prueba = [
        {
            "num": 1,
            "detalle": "Ruta directa desde recepción hasta el estante principal de cascos y visores",
            "inicio": CasillaBodega(0, 0),
            "meta": CasillaBodega(2, 4)
        },
        {
            "num": 2,
            "detalle": "Ruta de abastecimiento evitando el pasillo congestionado del taller de servicio",
            "inicio": CasillaBodega(0, 0),
            "meta": CasillaBodega(5, 3)
        },
        {
            "num": 3,
            "detalle": "Recorrido logístico desde el fondo de la bodega hasta la zona de despacho de repuestos",
            "inicio": CasillaBodega(5, 0),
            "meta": CasillaBodega(0, 5)
        }
    ]

    lineas_md = [
        "# Semana 04 - Marco Tecnológico de la IA: Búsqueda A* y Algoritmos Adversariales",
        "## Proyecto: Monitoreo de Inventario Visual para Tienda de Motos",
        "",
        "### A. Descripción del Problema",
        "En la gestión de una tienda de repuestos para motocicletas, el tiempo de respuesta del operario al buscar piezas solicitadas (como kits de arrastre, bujías o pastillas de freno) is crítico para mantener la operatividad comercial. El sistema utiliza búsqueda heurística para determinar la ruta de recolección (*picking*) más corta dentro de la bodega, evitando estanterías bloqueadas y penalizando zonas de congestión (como el área del taller).",
        "",
        "### B. Definición del Problema en Espacio de Estados",
        "- **Estado Inicial:** Coordenada $(fila, columna)$ donde se ubica el operario al recibir la orden.",
        "- **Espacio de Estados:** Todas las casillas transitables $(fila, columna)$ dentro de la cuadrícula de la bodega.",
        "- **Acciones / Operadores:** Desplazamiento ortogonal (arriba, abajo, izquierda, derecha).",
        "- **Transición:** Cambio de posición a una celda vecina si no constituye un obstáculo físico (estantería).",
        "- **Meta:** Coordenada $(fila, columna)$ exacta del estante donde se almacena el repuesto requerido.",
        "- **Costo de Camino $g(n)$:** Variable según el terreno (1.0 para pasillos libres, 3.0 para zonas de alta congestión).",
        "- **Heurística $h(n)$:** Distancia Manhattan, óptima para espacios en cuadrícula sin movimientos diagonales.",
        "- **Función de Prioridad:** $f(n) = g(n) + h(n)$.",
        "",
        "### C. Pruebas de Ejecución (Algoritmo A*)",
    ]

    print("=" * 80)
    print("SEMANA 04 - PRUEBAS DE RUTA ÓPTIMA (BÓDEGA DE REPUESTOS)")
    print("=" * 80)

    for prueba in casos_de_prueba:
        camino, costo, pasos = buscar_ruta_a_estrella(mapa_bodega, prueba["inicio"], prueba["meta"])
        texto_camino = " -> ".join([f"({c.fila},{c.columna})" for c in camino]) if camino else "Sin Ruta"

        print(f"\nCaso {prueba['num']}: {prueba['detalle']}")
        print(
            f"  -- Inicio: ({prueba['inicio'].fila}, {prueba['inicio'].columna}) | Meta: ({prueba['meta'].fila}, {prueba['meta'].columna})")
        print(f"  -- Costo total acumulado g(n): {costo}")
        print(f"  -- Nodos evaluados: {pasos}")
        print(f"  -- Ruta calculada: {texto_camino}")

        lineas_md.extend([
            f"#### Prueba {prueba['num']}: {prueba['detalle']}",
            f"- **Trayecto:** Desde `({prueba['inicio'].fila},{prueba['inicio'].columna})` hasta `({prueba['meta'].fila},{prueba['meta'].columna})`.",
            f"- **Ruta Calculada:** `{texto_camino}`",
            f"- **Costo Total $g(n)$:** `{costo}` | **Nodos Evaluados:** `{pasos}`",
            f"- **Justificación:** El algoritmo equilibra la distancia real y los costos de congestión del terreno para trazar el camino más eficiente sin colisionar con el inventario físico.",
            ""
        ])

    # Prueba de Minimax con Poda Alfa-Beta
    valores_ejemplo = [3, 5, 2, 9, 12, 5, 23, 23]
    resultado_minimax = probar_minimax_alfa_beta(0, 0, True, valores_ejemplo, -math.inf, math.inf)

    print("\n" + "=" * 80)
    print("PRUEBA DE DECISIÓN ADVERSARIAL (MINIMAX CON PODA ALFA-BETA)")
    print("=" * 80)
    print(f"Resultado óptimo evaluado: {resultado_minimax}")

    lineas_md.extend([
        "### D. Análisis de Decisiones Adversariales (Minimax y Poda Alfa-Beta)",
        "- **Aplicabilidad al Proyecto:** Los estantes y componentes de la tienda son pasivos y no actúan como adversarios racionales. Sin embargo, el marco teórico de Minimax se modela para escenarios de negociación automatizada con proveedores de repuestos o asignación competitiva de recursos.",
        f"- **Resultado de Simulación:** En un árbol de decisión de profundidad 3, el algoritmo evaluó el escenario óptimo devolviendo un valor de utilidad de `{resultado_minimax}`.",
        "- **Eficiencia (Poda Alfa-Beta):** Permitió descartar ramas de decisión redundantes que no afectaban el resultado final, reduciendo el tiempo de cómputo.",
        "",
        "### E. Conclusiones y Trade-offs",
        "- **Optimalidad:** $A^*$ garantiza encontrar la ruta de menor costo en la bodega gracias a la admisibilidad de la heurística de Manhattan.",
        "- **Rendimiento:** El uso de colas de prioridad (*heapq*) optimiza la selección del nodo prometedor en tiempo logarítmico.",
    ])

    RUTA_REPORTE.write_text("\n".join(lineas_md), encoding="utf-8")
    print(f"\nReporte generado exitosamente en: {RUTA_REPORTE}\n")


if __name__ == "__main__":
    correr_pruebas_y_generar_informe()