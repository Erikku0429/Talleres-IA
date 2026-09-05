import os
import math
from collections import Counter
from pathlib import Path

DIRECTORIO_RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DATA = DIRECTORIO_RAIZ / "data"
CARPETA_REPORTES = DIRECTORIO_RAIZ / "reports"

RUTA_BASE_CONOCIMIENTO = CARPETA_DATA / "base_conocimiento.txt"
RUTA_REPORTE = CARPETA_REPORTES / "semana05.md"


# BASE DE CONOCIMIENTO

BASE_CONOCIMIENTO_TEXTO = """1. Los kits de arrastre (cadena, piñón y corona) para motos de 200cc a 400cc requieren revisión de stock semanal y reorden inmediata cuando queden menos de 3 unidades.
2. Las pastillas de freno cerámicas y orgánicas para Bajaj Dominar 400 y KTM Duke tienen alta rotación por desgaste preventivo en mantenimientos de 5000 km.
3. El aceite sintético 10W-50 y 15W-50 para motor de cuatro tiempos presenta sobrestock si el inventario supera las 50 latas sin rotación en 30 días.
4. Los discos de embrague y guayas de acelerador para motocicletas de mensajería urbana sufren falla recurrente por tráfico pesado y uso intensivo.
5. Los llantas de compuesto blando y semislick para rodadas en carretera requieren almacenamiento en espacio seco y temperatura controlada para evitar cristalización.
6. El sistema eléctrico, baterías de gel y bujías de iridio muestran pico de demanda durante temporadas de lluvia e inundación vial.
7. Los repuestos descontinuados o de bajo movimiento (carburadores antiguos, plásticos de modelos >10 años) deben catalogarse para remate o devolución a proveedor.
8. El kit de empaquetadura de motor y retenes de suspensión delantera son ítems críticos en mantenimiento mayor de talleres aliados.
"""

def asegurar_base_conocimiento():
    CARPETA_DATA.mkdir(parents=True, exist_ok=True)
    if not RUTA_BASE_CONOCIMIENTO.exists():
        RUTA_BASE_CONOCIMIENTO.write_text(BASE_CONOCIMIENTO_TEXTO, encoding="utf-8")

def cargar_base_conocimiento():
    asegurar_base_conocimiento()
    lineas = RUTA_BASE_CONOCIMIENTO.read_text(encoding="utf-8").strip().split("\n")
    return [linea.strip() for linea in lineas if linea.strip()]

# MOTOR DE SISTEMA EXPERTO
class SistemaExpertoInventario:
    """
    Evaluación de reglas condicionales basadas en eventos e indicadores
    de inventario de la tienda de repuestos.
    """
    def evaluar(self, consulta: str) -> str:
        consulta_lower = consulta.lower()

        if ("kit" in consulta_lower or "arrastre" in consulta_lower or "cadena" in consulta_lower) and ("bajo" in consulta_lower or "agotado" in consulta_lower or "urgente" in consulta_lower):
            return "REGLA_1_ACTIVADA: [REORDEN URGENTE] Emitir orden de compra inmediata a proveedor principal de kits de arrastre."

        if ("pastillas" in consulta_lower or "freno" in consulta_lower or "mantenimiento" in consulta_lower) and ("revisión" in consulta_lower or "desgaste" in consulta_lower or "5000" in consulta_lower):
            return "REGLA_2_ACTIVADA: [ALTA ROTACIÓN] Incrementar el stock de seguridad un 25% para pastillas de freno en zona de taller."

        if ("aceite" in consulta_lower or "sintético" in consulta_lower or "lubricante" in consulta_lower) and ("exceso" in consulta_lower or "alto" in consulta_lower or "sin rotación" in consulta_lower or "sobrestock" in consulta_lower):
            return "REGLA_3_ACTIVADA: [PROMOCIÓN / REDISTRIBUCIÓN] Lanzar combo promocional de cambio de aceite con filtro o mover a bodega secundaria."

        if ("batería" in consulta_lower or "eléctrico" in consulta_lower or "bujía" in consulta_lower or "lluvia" in consulta_lower):
            return "REGLA_4_ACTIVADA: [ALERTA CLIMÁTICA] Verificar reserva estratégica de baterías de gel y componentes eléctricos ante lluvias."

        if ("descontinuado" in consulta_lower or "antiguo" in consulta_lower or "obsoleto" in consulta_lower or "remate" in consulta_lower):
            return "REGLA_5_ACTIVADA: [DEPURACIÓN DE INVENTARIO] Aplicar descuento de liquidación o gestionar devolución a proveedor."

        return "REGLA_DEFAULT: [MONITOREO ESTÁNDAR] Registrar solicitud en el log operativo y verificar niveles mínimos en sistema ERP."

# TF-IDF Y SIMILITUD COSENO
class RecuperadorTFIDF:
    def __init__(self, documentos: list[str]):
        self.documentos = documentos
        self.doc_tokens = [self._tokenizar(doc) for doc in documentos]
        self.vocabulario = list(set(word for doc in self.doc_tokens for word in doc))
        self.num_docs = len(documentos)
        self.idf = self._calcular_idf()
        self.matriz_tfidf = [self._vector_tfidf(tokens) for tokens in self.doc_tokens]

    def _tokenizar(self, texto: str) -> list[str]:
        palabras_ignorar = {"el", "la", "los", "las", "un", "una", "de", "en", "para", "con", "y", "o", "por", "a", "del", "que"}
        tokens = texto.lower().replace(".", "").replace(",", "").replace("(", "").replace(")", "").split()
        return [t for t in tokens if t not in palabras_ignorar and len(t) > 2]

    def _calcular_idf(self) -> dict[str, float]:
        idf = {}
        for palabra in self.vocabulario:
            docs_con_palabra = sum(1 for doc in self.doc_tokens if palabra in doc)
            idf[palabra] = math.log((1 + self.num_docs) / (1 + docs_con_palabra)) + 1
        return idf

    def _vector_tfidf(self, tokens: list[str]) -> list[float]:
        tf = Counter(tokens)
        total_tokens = len(tokens) if len(tokens) > 0 else 1
        vector = []
        for palabra in self.vocabulario:
            tf_val = tf[palabra] / total_tokens
            vector.append(tf_val * self.idf[palabra])
        return vector

    def _similitud_coseno(self, vec1: list[float], vec2: list[float]) -> float:
            dot = sum(a * b for a, b in zip(vec1, vec2))
            norm1 = math.sqrt(sum(a * a for a in vec1)) if any(vec1) else 0.0
            norm2 = math.sqrt(sum(b * b for b in vec2)) if any(vec2) else 0.0
            if norm1 == 0 or norm2 == 0:
                return 0.0
            return dot / (norm1 * norm2)

    def buscar(self, consulta: str):
        tokens_consulta = self._tokenizar(consulta)
        vec_consulta = self._vector_tfidf(tokens_consulta)
        
        similitudes = []
        for i, vec_doc in enumerate(self.matriz_tfidf):
            sim = self._similitud_coseno(vec_consulta, vec_doc)
            similitudes.append((i, sim))
        
        similitudes.sort(key=lambda x: x[1], reverse=True)
        mejor_idx, mejor_sim = similitudes[0]
        return self.documentos[mejor_idx], mejor_sim

# CLASIFICACIÓN 

DATOS_ENTRENAMIENTO_CLASIFICADOR = [
    
    ("Stock agotado de kit de arrastre para Dominar 400", "RIESGO_REORDEN"),
    ("Faltan pastillas de freno en el inventario del taller", "RIESGO_REORDEN"),
    ("Se agotaron las cadenas y piñones de repuesto urgente", "RIESGO_REORDEN"),
    ("Nivel crítico de existencias de bujías e insumos", "RIESGO_REORDEN"),
    ("Sin disponibilidad de kit de empaquetadura de motor", "RIESGO_REORDEN"),

    ("Demasiadas latas de aceite 10W50 aculadas en bodega", "SOBRESTOCK"),
    ("Exceso de llantas semislick almacenadas sin venta", "SOBRESTOCK"),
    ("Sobrestock de cascos y accesorios de temporada pasada", "SOBRESTOCK"),
    ("Inventario paralizado de lubricantes sin rotación en 30 días", "SOBRESTOCK"),

    ("Batería de gel defectuosa o sulfatada por almacenamiento", "INCIDENTE_CALIDAD"),
    ("Pastillas de freno desgastadas prematuramente por tráfico", "INCIDENTE_CALIDAD"),
    ("Llantas cristalizadas por mal almacenamiento en bodega", "INCIDENTE_CALIDAD"),

    ("Carburador antiguo de modelo descontinuado hace 10 años", "OBSOLESCENCIA"),
    ("Repuestos obsoletos de motocicletas fuera de mercado", "OBSOLESCENCIA"),
    ("Plásticos y carenajes antiguos para remate de inventario", "OBSOLESCENCIA")
]

class ClasificadorNaiveBayes:
    def __init__(self, datos_entrenamiento):
        self.clases = set(c for _, c in datos_entrenamiento)
        self.vocabulario = set()
        self.conteo_palabras = {c: Counter() for c in self.clases}
        self.total_palabras_clase = {c: 0 for c in self.clases}
        self.conteo_docs_clase = {c: 0 for c in self.clases}
        self.total_docs = len(datos_entrenamiento)

        for texto, clase in datos_entrenamiento:
            self.conteo_docs_clase[clase] += 1
            tokens = self._tokenizar(texto)
            for token in tokens:
                self.vocabulario.add(token)
                self.conteo_palabras[clase][token] += 1
                self.total_palabras_clase[clase] += 1

    def _tokenizar(self, texto: str) -> list[str]:
     return [p.lower() for p in texto.replace(",", "").replace(".", "").split() if len(p) > 2]

    def clasificar(self, consulta: str) -> str:
        tokens = self._tokenizar(consulta)
        V = len(self.vocabulario)
        mejores_score = {}

        for clase in self.clases:
            # Prior log P(C)
            prior = math.log(self.conteo_docs_clase[clase] / self.total_docs)
            likelihood = 0.0
            for token in tokens:
                # Laplace Smoothing (+1)
                count = self.conteo_palabras[clase][token]
                prob = (count + 1) / (self.total_palabras_clase[clase] + V)
                likelihood += math.log(prob)
            mejores_score[clase] = prior + likelihood

        return max(mejores_score, key=mejores_score.get)


def ejecutar_sistema_hibrido():
    base_conocimiento = cargar_base_conocimiento()
    experto = SistemaExpertoInventario()
    recuperador = RecuperadorTFIDF(base_conocimiento)
    clasificador = ClasificadorNaiveBayes(DATOS_ENTRENAMIENTO_CLASIFICADOR)

    consultas_prueba = [
        "El kit de arrastre para Dominar 400 está bajo en stock y se requiere reorden urgente.",
        "Hay un exceso de aceite sintético 10W-50 acumulado en bodega sin rotación en 30 días.",
        "Se detectaron fallas en el sistema eléctrico y baterías de gel debido a la temporada de lluvia."
    ]

    resultados_pruebas = []

    print("==========================================================")
    print("SISTEMA HÍBRIDO DE MONITOREO DE INVENTARIO (SEMANA 05)")
    print("==========================================================")

    for idx, consulta in enumerate(consultas_prueba, start=1):
        regla = experto.evaluar(consulta)
        doc_recuperado, similitud = recuperador.buscar(consulta)
        categoria = clasificador.clasificar(consulta)

        res = {
            "num": idx,
            "consulta": consulta,
            "regla": regla,
            "doc_recuperado": doc_recuperado,
            "similitud": round(similitud, 4),
            "categoria": categoria
        }
        resultados_pruebas.append(res)

        print(f"\n--- CONSULTA DE PRUEBA #{idx} ---")
        print(f"Consulta: '{consulta}'")
        print(f"Regla Activada: {regla}")
        print(f"Información Recuperada: {doc_recuperado}")
        print(f"Valor de Similitud (TF-IDF): {res['similitud']}")
        print(f"Clasificación Predicha: {categoria}")

    generar_reporte_md(resultados_pruebas)

def generar_reporte_md(resultados):
    CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)
    
    lineas = [
        "# Semana 05 - Sistema Híbrido: Sistemas Expertos, TF-IDF y Clasificación",
        "**Proyecto:** Sistema de Monitoreo de Inventario para Tienda de Motos  ",
        "**Autores:** Erick Santiago Garcia Sanchez & Oscar David Gutierrez  ",
        "",
        "## 1. Descripción del Sistema Híbrido",
        "El sistema combina tres componentes fundamentales de la Inteligencia Artificial explicable para la gestión de inventarios:",
        "1. **Motor de Reglas Explicables (Sistema Experto):** Evalúa disparadores de negocio (reorden, promociones, alertas climáticas).",
        "2. **Recuperador de Información (TF-IDF + Similitud Coseno):** Mapea consultas contra la base de conocimiento estructurada (`data/base_conocimiento.txt`).",
        "3. **Clasificador Naive Bayes Supervisado:** Categoriza automáticamente la intención de la solicitud en etiquetas operativas (`RIESGO_REORDEN`, `SOBRESTOCK`, `INCIDENTE_CALIDAD`, `OBSOLESCENCIA`).",
        "",
        "## 2. Base de Conocimiento Cargada",
        "Se registraron 8 entradas de dominio específicas para la tienda y taller de motocicletas:",
        ""
    ]

    base_conocimiento = cargar_base_conocimiento()
    for entrada in base_conocimiento:
        lineas.append(f"- {entrada}")

    lineas.extend([
        "",
        "## 3. Pruebas de Ejecución y Resultados Explicables",
        "",
        "| Consulta | Regla Activada | Evidencia Recuperada | Similitud | Clasificación |",
        "| :--- | :--- | :--- | :---: | :---: |"
    ])

    for r in resultados:
        doc_corto = r['doc_recuperado'][:60] + "..." if len(r['doc_recuperado']) > 60 else r['doc_recuperado']
        regla_corta = r['regla'].split(":")[0]
        lineas.append(f"| {r['consulta']} | `{regla_corta}` | {doc_corto} | `{r['similitud']}` | **{r['categoria']}** |")

    lineas.extend([
        "",
        "### Detalle Explicativo de las Consultas",
        ""
    ])

    for r in resultados:
        lineas.extend([
            f"#### Consulta #{r['num']}: \"{r['consulta']}\"",
            f"- **Regla Activada:** {r['regla']}",
            f"- **Evidencia Recuperada:** {r['doc_recuperado']}",
            f"- **Similitud TF-IDF:** `{r['similitud']}`",
            f"- **Clasificación Predicha:** `{r['categoria']}`",
            f"- **Explicación del Resultado:** El sistema identificó palabras clave relativas al estado del ítem y aplicó la regla lógica correspondiente. Al mismo tiempo, la búsqueda TF-IDF aisló la norma de inventario con mayor coincidencia léxico-semántica y Naive Bayes determinó la categoría operativa según los 15 ejemplos de entrenamiento etiquetados.",
            ""
        ])

    lineas.extend([
        "## 4. Conclusión",
        "El sistema híbrido cumple exitosamente con la integración requerida para la Semana 5. Permite tomar decisiones automáticas explicables y proporciona soporte de trazabilidad para la gestión de repuestos de motocicletas."
    ])

    RUTA_REPORTE.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\n[OK] Reporte generado exitosamente en: {RUTA_REPORTE}")

if __name__ == "__main__":
    ejecutar_sistema_hibrido()
