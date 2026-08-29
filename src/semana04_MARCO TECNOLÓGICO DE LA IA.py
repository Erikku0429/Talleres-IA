"""
Proyecto: Monitoreo de Inventario para Tienda de Motos
Ingeniería de Sistemas S10A
Oscar David Gutierrez, Erick Santiago Garcia
"""

import math
import heapq
from dataclasses import dataclass
from pathlib import Path

# Configuración de carpetas del proyecto
DIRECTORIO_RAIZ = Path(__file__).resolve().parent.parent
CARPETA_REPORTES = DIRECTORIO_RAIZ / "reports"
RUTA_REPORTE = CARPETA_REPORTES / "semana04.md"

# 1. BÚSQUEDA A*: RUTA ÓPTIMA

@dataclass(frozen=True)
class CasillaBodega:
    fila: int
    columna: int

    def __lt__(self, otra):
        return (self.fila, self.columna) < (otra.fila, otra.columna)

class MapaBodega:
    """
    Representa el mapa de la bodega como una cuadrícula de casillas:
    0 = Pasillo despejado
    1 = Obstáculo
    2 = Zona concurrida
    """
    def __init__(self, croquis: list[list[int]]):
        self.croquis = croquis
        self.filas = len(croquis)
        self.columnas = len(croquis[0])

    def es_paso_valido(self, casilla: CasillaBodega) -> bool:
        # Revisa que no se salga y a su vez no hayan obstaculos
        dentro_del_mapa = 0 <= casilla.fila < self.filas and 0 <= casilla.columna < self.columnas
        return dentro_del_mapa and self.croquis[casilla.fila][casilla.columna] != 1

    def obtener_costo_terreno(self, casilla: CasillaBodega) -> float:
        # Se validan los costos, Caminar por pasillo normal cuesta 1, por zona de tráfico cuesta 3
        return 3.0 if self.croquis[casilla.fila][casilla.columna] == 2 else 1.0

    def obtener_casillas_vecinas(self, casilla: CasillaBodega) -> list[CasillaBodega]:
        # Permite moverse a los lados, arriba o abajo
        pasos_posibles = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        vecinos = []
        for df, dc in pasos_posibles:
            siguiente = CasillaBodega(casilla.fila + df, casilla.columna + dc)
            if self.es_paso_valido(siguiente):
                vecinos.append(siguiente)
        return vecinos

def calcular_distancia_linea_recta(origen: CasillaBodega, destino: CasillaBodega) -> float:
    # Heurística h(n): Distancia recta en la cuadrícula (Euclidiana)
    return math.sqrt((origen.fila - destino.fila)**2 + (origen.columna - destino.columna)**2)

def buscar_ruta_a_estrella(mapa: MapaBodega, punto_inicio: CasillaBodega, punto_meta: CasillaBodega):
    """
    Algoritmo A* para encontrar la mejor ruta de recolección de repuestos.
    Calcula: f(n) = g(n) + h(n)
    g(n) -> Distancia/costo recorrido hasta el momento.
    h(n) -> Estimación de lo que falta para llegar a la meta.
    """
    cola_prioridad = []
    heapq.heappush(cola_prioridad, (0.0, punto_inicio))
    
    rastreo_camino = {}
    costo_recorrido_g = {punto_inicio: 0.0}
    
    pasos_evaluados = 0

    while cola_prioridad:
        f_actual, casilla_actual = heapq.heappop(cola_prioridad)
        pasos_evaluados += 1

        # Si llegamos a la estantería buscada, armamos la ruta final
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
                estimacion_f = nuevo_costo_g + calcular_distancia_linea_recta(vecina, punto_meta)
                heapq.heappush(cola_prioridad, (estimacion_f, vecina))

    return None, float('inf'), pasos_evaluados


# 2. PRÁCTICA DE REFERENCIA: MINIMAX CON PODA ALFA-BETA

def probar_minimax_alfa_beta(nodo: int, nivel: int, turno_max: bool, valores: list[int], alfa: float, beta: float) -> int:
    # Aca simulamos el peor escenario posible en cuanto a toma de desiciones se refiere
    if nivel == 3:
        return valores[nodo]

    if turno_max:
        mejor_opcion = -math.inf
        for i in range(2):
            puntaje = probar_minimax_alfa_beta(nodo * 2 + i, nivel + 1, False, valores, alfa, beta)
            mejor_opcion = max(mejor_opcion, puntaje)
            alfa = max(alfa, mejor_opcion)
            if beta <= alfa:
                break # Aca borramos las ramas que no tiene sentido revisar 
        return mejor_opcion
    else:
        mejor_opcion = math.inf
        for i in range(2):
            puntaje = probar_minimax_alfa_beta(nodo * 2 + i, nivel + 1, True, valores, alfa, beta)
            mejor_opcion = min(mejor_opcion, puntaje)
            beta = min(beta, mejor_opcion)
            if beta <= alfa:
                break # Aca borramos las ramas que no tiene sentido revisar 
        return mejor_opcion


# 3. Se genera el reporte y se realizan las pruebas

def correr_pruebas_y_generar_informe():
    CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)

    # Plano de la bodega (6x6)
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
            "detalle": "Ir directo desde recepción hasta el estante de cascos",
            "inicio": CasillaBodega(0, 0),
            "meta": CasillaBodega(2, 4)
        },
        {
            "num": 2,
            "detalle": "Ruta evitando el paso congestionado del taller de mecánicos",
            "inicio": CasillaBodega(0, 0),
            "meta": CasillaBodega(5, 3)
        },
        {
            "num": 3,
            "detalle": "Recorrido largo desde el fondo de la bodega a la zona de despacho",
            "inicio": CasillaBodega(5, 0),
            "meta": CasillaBodega(0, 5)
        }
    ]

    lineas_md = [
        "Semana 04 - Marco Tecnológico de la IA: Búsqueda A* y Algoritmos Adversariales",
        "Proyecto: Sistema de Monitoreo de Inventario para Tienda de Motos",
        "",
        "A. Descripción del Problema",
        "En la bodega de repuestos necesitamos que el operario encuentre la forma más rápida de ir por las piezas pedidas (como kits de arrastre, aceite o pastillas de freno)."
        "El objetivo es armar la ruta más corta evitando estantes bloqueados y pasillos llenos de gente.",
        "",
        "B. Definición del Problema en Estados",
        " --Estado Inicial: Coordenada filas,columnas donde arranca el operario.",
        " --Posibles Estados: Cualquier casilla caminable filas,columnas dentro de la bodega.",
        " --Acciones: Moverse un paso hacia arriba,abajo,izquierda,derecha.",
        " --Transiciones: Pasar a la casilla de al lado si no hay un muro o estante estorbando.",
        " --Meta: Coordenada fila_meta,columna_meta de la estantería del repuesto.",
        " --Costo de Camino g(n): Se validan los costos de cada accion, pasillo libre 1.0 y zona concurrida 3.0",
        " --Heurística h(n): Distancia Euclidiana (línea recta) desde la casilla actual hasta la meta.",
        " --Criterio de Selección: Escoger la ruta que dé el menor resultado en $f(n) = g(n) + h(n)$.",
        "",
        "C. Pruebas de Ejecución",
    ]

    
    print("SEMANA 04 - PRUEBAS DE RUTA ÓPTIMA")
   

    for prueba in casos_de_prueba:
        camino, costo, pasos = buscar_ruta_a_estrella(mapa_bodega, prueba["inicio"], prueba["meta"])
        texto_camino = " -> ".join([f"({c.fila},{c.columna})" for c in camino]) if camino else "Sin Ruta"
        
        print(f"\nCaso {prueba['num']}: {prueba['detalle']}")
        print(f"--Inicio: ({prueba['inicio'].fila}, {prueba['inicio'].columna}) | Meta: ({prueba['meta'].fila}, {prueba['meta'].columna})")
        print(f"--Costo total g(n): {costo}")
        print(f"--Casillas evaluadas: {pasos}")
        print(f"--Ruta seguida: {texto_camino}")

        lineas_md.extend([
            f" --Prueba {prueba['num']}: {prueba['detalle']}",
            f" --Entrada: Desde `({prueba['inicio'].fila},{prueba['inicio'].columna})` hasta `({prueba['meta'].fila},{prueba['meta'].columna})`.",
            f" --Ruta Calculada: `{texto_camino}`",
            f" --Costo final $g(n)$: `{costo}` | Casillas revisadas: `{pasos}`",
            f" --Explicación**: El algoritmo evalúa el mapa y selecciona el camino que evita estantes y zonas de tráfico, equilibrando la distancia recorrida con lo que falta para llegar.",
            ""
        ])

    # Prueba de Minimax
    valores_ejemplo = [3, 5, 2, 9, 12, 5, 23, 23]
    resultado_minimax = probar_minimax_alfa_beta(0, 0, True, valores_ejemplo, -math.inf, math.inf)
    
    
    print("SEMANA 04 - PRUEBA DE MINIMAX Y PODA ALFA-BETA")
   
    print(f"Resultado devuelto por Minimax: {resultado_minimax}")

    lineas_md.extend([
        "D. Resultados",
        "--¿Aplica al proyecto?: No aplica, porque los estantes no hacen daño al negocio.",
        "--Diferencia técnica: La búsqueda $A^*$ sirve para encontrar una ruta eficiente. Minimax se usa cuando hay un rival pensante tratando de contrarrestar nuestras decisiones.",
        "--Poda Alfa-Beta: Este ayuda a quitar opciones que sabemos que no se van a tomar",
        "--Prueba realizada: En un árbol de ejemplo de 3 niveles, el algoritmo seleccionó correctamente la mejor decisión con valor de `" + str(resultado_minimax) + "`.",
        "",
        "E. Análisis",  
        "--Lo bueno: $A^*$ siempre encuentra la ruta más rápida si la estimación de distancia no exagera el costo real.",
        "--Por mejorar: hasta el momento todo funciona con datos establecidos",
    ])

    RUTA_REPORTE.write_text("\n".join(lineas_md), encoding="utf-8")
    print(f"\nInforme generado correctamente en: {RUTA_REPORTE}\n")

if __name__ == "__main__":
    correr_pruebas_y_generar_informe()