import math
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent if "__file__" in locals() else Path(".")
REP_DIR = ROOT / "reports"
REP_DIR.mkdir(parents=True, exist_ok=True)


def distancia_euclidiana(vec1: list[float], vec2: list[float]) -> float:
    return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))


def probar_representacion_numerica():
    pieza_referencia = [2.0, 35.0]  # Stock crítico (2 unds), 35 días sin rotación
    pieza_prueba1 = [1.0, 40.0]  # Estado crítico similar
    pieza_prueba2 = [45.0, 2.0]  # Stock alto, alta rotación

    d1 = distancia_euclidiana(pieza_referencia, pieza_prueba1)
    d2 = distancia_euclidiana(pieza_referencia, pieza_prueba2)
    return pieza_referencia, pieza_prueba1, pieza_prueba2, d1, d2


def evaluar_regla_simbolica(stock: int, dias_rotacion: int, oxidacion: bool) -> str:
    hechos = {
        "stock": stock,
        "dias_rotacion": dias_rotacion,
        "oxidacion": oxidacion
    }

    if hechos["stock"] < 3 or hechos["dias_rotacion"] > 30 or hechos["oxidacion"]:
        conclusion = "SI se detecta stock bajo o deterioro (oxidación/estancamiento) -> ENTONCES generar orden de compra urgente o activar protocolo de descarte."
    else:
        conclusion = "SI el inventario es óptimo -> ENTONCES mantener monitoreo estándar en ERP."
    return conclusion


class AutomataInventarioMotos:

    def __init__(self):
        self.estado_inicial = "S0"
        self.estados_aceptacion = {"S4"}

    def procesar_secuencia(self, secuencia: str) -> bool:
        estado_actual = self.estado_inicial

        for char in secuencia:
            if estado_actual == "S0" and char == 'M':
                estado_actual = "S1"
            elif estado_actual == "S1" and char == 'O':
                estado_actual = "S2"
            elif estado_actual == "S2" and char == 'T':
                estado_actual = "S3"
            elif estado_actual == "S3" and char.isdigit():
                estado_actual = "S4"
            else:
                return False
        return estado_actual in self.estados_aceptacion


# 4. ORQUESTACIÓN Y GENERACIÓN DE REPORTE
def ejecutar_semana_07():
    print("=" * 80)
    print("SEMANA 07 - REPRESENTACIONES DEL RECONOCIMIENTO (TIENDA DE MOTOS)")
    print("=" * 80)

    ref, p1, p2, d1, d2 = probar_representacion_numerica()
    print(f"1. Numérico -> Distancia a pieza crítica 1: {d1:.2f}, Distancia a pieza 2: {d2:.2f}")

    conclusion = evaluar_regla_simbolica(stock=1, dias_rotacion=40, oxidacion=False)
    print(f"2. Simbólico -> {conclusion}")

    afd = AutomataInventarioMotos()
    valido = afd.procesar_secuencia("MOT5")
    invalido = afd.procesar_secuencia("MOTO")
    print(f"3. Autómata -> Secuencia 'MOT5' aceptada: {valido} | Secuencia 'MOTO' aceptada: {invalido}")

    reporte_path = REP_DIR / "semana07.md"
    contenido_md = f"""# Semana 07 - Representaciones del Reconocimiento
## Proyecto: Monitoreo de Inventario Visual y Gestión para Tienda de Motos

### 1. Introducción
En esta práctica se implementaron y evaluaron tres representaciones del reconocimiento aplicadas a la gestión de inventario de repuestos para motocicletas:
1. **Representación Numérica:** Análisis vectorial de características de stock y distancia euclidiana.
2. **Representación Simbólica:** Motor de reglas lógicas para toma de decisiones en taller.
3. **Autómata Finito:** Validación formal de códigos y patrones de partes en el sistema.

### 2. Implementación de las Representaciones

#### A. Representación Numérica
Se define un vector de características compuesto por `[nivel_stock, dias_sin_rotacion]`. Utilizando la distancia euclidiana, el sistema compara piezas para identificar similitudes en estado crítico.
- **Pieza de Referencia (Crítica):** `{ref}`
- **Distancia a Pieza 1 (`{p1}`):** `{d1:.2f}` (Alta similitud, requiere atención).
- **Distancia a Pieza 2 (`{p2}`):** `{d2:.2f}` (Baja similitud, inventario saludable).

#### B. Representación Simbólica
Se establecieron hechos operativos y reglas de negocio adaptadas al almacén de repuestos:
- **Hechos:** Stock actual, días de estancamiento y estado de oxidación de la pieza.
- **Regla Aplicada:** 
  > *SI* stock < 3 *O* días_rotacion > 30 *O* oxidación == True $\rightarrow$ *ENTONCES* generar orden de compra urgente o activar protocolo de descarte.
- **Resultado de Evaluación de Prueba:** {conclusion}

### C. Reconocimiento mediante Autómata Finito
Se diseñó un Autómata Finito Determinista (AFD) para validar los códigos alfanuméricos de control de inventario de las autopartes (ej. formato `MOT` seguido de un dígito de validación).
- **Alfabeto ($\Sigma$):** `{{'M', 'O', 'T', [0-9]}}`
- **Estados ($Q$):** `{{'S0', 'S1', 'S2', 'S3', 'S4'}}`
- **Estado Inicial:** `S0`
- **Estados de Aceptación:** `{{'S4'}}`
- **Pruebas:** 
  - Secuencia `"MOT5"` $\rightarrow$ **Aceptada** (`{valido}`).
  - Secuencia `"MOTO"` $\rightarrow$ **Rechazada** (`{invalido}`).

### 3. Tabla Comparativa de Representaciones

| Representación | Qué información utiliza | Qué puede reconocer | Ventajas | Limitations | Información que puede perderse |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Numérica** | Vectores de valores cuantitativos (stock, días, costos). | Grados de similitud y proximidad matemática entre repuestos. | Permite cálculos precisos y clasificación estadística rápida. | Sensible a escalas y unidades de medida heterogéneas. | Contexto cualitativo o descripciones en lenguaje natural. |
| **Simbólica** | Hechos discretos y reglas condicionales lógicas. | Causalidad y directivas normativas del negocio de autopartes. | Alta explicabilidad y alineación directa con expertos humanos. | Dificultad para manejar incertidumbre o datos continuos masivos. | Matices graduales o probabilidades estadísticas de fallo. |
| **Autómata** | Secuencias formales de símbolos y alfabetos definidos. | Patrones sintácticos exactos y códigos de control válidos. | Determinista, eficiente en tiempo de ejecución y fácil de verificar. | Rigidez absoluta ante variaciones o errores tipográficos. | Estructuras semánticas o relaciones contextuales amplias. |

### 4. Conclusiones
La combinación de las tres representaciones demuestra cómo un sistema inteligente puede abordar el control de inventarios desde diferentes perspectivas: matemática, lógica experta y validación formal de patrones.
"""

    reporte_path.write_text(contenido_md, encoding="utf-8")
    print(f"\n[OK] Reporte generado exitosamente en: {reporte_path}")


if __name__ == "__main__":
    ejecutar_semana_07()