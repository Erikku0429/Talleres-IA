import os
import math
from collections import Counter
from pathlib import Path

DIRECTORIO_RAIZ = Path(__file__).resolve().parent.parent
CARPETA_DATA = DIRECTORIO_RAIZ / "data"
CARPETA_REPORTES = DIRECTORIO_RAIZ / "reports"

RUTA_BASE_CONOCIMIENTO = CARPETA_DATA / "base_conocimiento.txt"
RUTA_REPORTE = CARPETA_REPORTES / "semana05.md"


# 1. CARGA DE LA BASE DE CONOCIMIENTO DESDE EL ARCHIVO TXT EXTERNO
def cargar_base_conocimiento() -> list[str]:
    if not RUTA_BASE_CONOCIMIENTO.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo de base de conocimiento en: {RUTA_BASE_CONOCIMIENTO}. "
            "Por favor, asegúrate de crear el archivo data/base_conocimiento.txt con las entradas requeridas."
        )

    lineas = RUTA_BASE_CONOCIMIENTO.read_text(encoding="utf-8").strip().split("\n")
    documentos = [linea.strip() for linea in lineas if linea.strip()]

    if len(documentos) < 8:
        raise ValueError(
            f"La base de conocimiento debe contener al menos 8 entradas. Actualmente tiene {len(documentos)}.")

    return documentos


# 2. MOTOR DE SISTEMA EXPERTO (REGLAS PROPIAS DE NEGOCIO)
class SistemaExpertoInventario:
    """
    Evalúa reglas condicionales basadas en eventos, umbrales e indicadores
    operativos del inventario de la tienda y taller de repuestos de motocicletas.
    """

    def evaluar(self, consulta: str) -> str:
        consulta_lower = consulta.lower()

        if any(w in consulta_lower for w in ["kit de arrastre", "cadena oxidada", "piñón", "rotura"]) and any(
                w in consulta_lower for w in ["bajo", "agotado", "urgente", "riesgo"]):
            return "REGLA_1_ACTIVADA: [REORDEN URGENTE / BLOQUEO DE SEGURIDAD] Emitir orden de compra inmediata o retirar pieza defectuosa del inventario."

        if any(w in consulta_lower for w in ["pastillas", "freno", "mantenimiento", "embrague"]):
            return "REGLA_2_ACTIVADA: [ALTA ROTACIÓN] Incrementar el stock de seguridad un 25% para repuestos de desgaste preventivo en taller."

        if any(w in consulta_lower for w in ["aceite", "sintético", "lubricante", "sobrestock", "exceso"]):
            return "REGLA_3_ACTIVADA: [PROMOCIÓN / REDISTRIBUCIÓN] Lanzar combo promocional de cambio de aceite con filtro o reubicar stock estancado."

        if any(w in consulta_lower for w in ["batería", "eléctrico", "bujía", "lluvia"]):
            return "REGLA_4_ACTIVADA: [ALERTA CLIMÁTICA] Verificar reserva estratégica de componentes eléctricos y baterías ante temporadas de precipitaciones."

        if any(w in consulta_lower for w in ["descontinuado", "antiguo", "obsoleto", "remate"]):
            return "REGLA_5_ACTIVADA: [DEPURACIÓN DE INVENTARIO] Aplicar descuento de liquidación o gestionar devolución (RMA) al proveedor."

        if any(w in consulta_lower for w in ["fuga", "aceite hidráulico", "amortiguador", "retén"]):
            return "REGLA_6_ACTIVADA: [GARANTÍA / TALLER] Aislar componente defectuoso y activar protocolo de revisión técnica."

        if any(w in consulta_lower for w in ["visor", "rayón", "casco", "estético"]):
            return "REGLA_7_ACTIVADA: [REETIQUETADO COMERCIAL] Ajustar precio por defecto estético superficial sin comprometer la seguridad."

        return "REGLA_DEFAULT: [MONITOREO ESTÁNDAR] Registrar solicitud en el log operativo y verificar niveles en sistema ERP de la tienda."


# 3. RECUPERACIÓN DE INFORMACIÓN (TF-IDF Y SIMILITUD COSENO)
class RecuperadorTFIDF:
    def __init__(self, documentos: list[str]):
        self.documentos = documentos
        self.doc_tokens = [self._tokenizar(doc) for doc in documentos]
        self.vocabulario = list(set(word for doc in self.doc_tokens for word in doc))
        self.num_docs = len(documentos)
        self.idf = self._calcular_idf()
        self.matriz_tfidf = [self._vector_tfidf(tokens) for tokens in self.doc_tokens]

    def _tokenizar(self, texto: str) -> list[str]:
        palabras_ignorar = {"el", "la", "los", "las", "un", "una", "de", "en", "para", "con", "y", "o", "por", "a",
                            "del", "que", "se", "su"}
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


# 4. CLASIFICACIÓN SUPERVISADA
DATOS_ENTRENAMIENTO_CLASIFICADOR = [
    # Categoría: RIESGO_REORDEN
    ("Stock agotado de kit de arrastre para Dominar 400", "RIESGO_REORDEN"),
    ("Faltan pastillas de freno en el inventario del taller", "RIESGO_REORDEN"),
    ("Se agotaron las cadenas y piñones de repuesto urgente", "RIESGO_REORDEN"),
    ("Nivel crítico de existencias de bujías e insumos", "RIESGO_REORDEN"),
    ("Sin disponibilidad de kit de empaquetadura de motor", "RIESGO_REORDEN"),
    ("Desabastecimiento total de filtros de aire en bodega principal", "RIESGO_REORDEN"),
    ("Urgente reposición de aceite de motor para mantenimiento", "RIESGO_REORDEN"),
    ("Inventario en cero de guayas de embrague para mensajería", "RIESGO_REORDEN"),
    ("Faltan discos de freno delanteros para modelos de alta cilindrada", "RIESGO_REORDEN"),
    ("Alerta por escasez de bujías de iridio en temporada alta", "RIESGO_REORDEN"),

    # Categoría: SOBRESTOCK
    ("Demasiadas latas de aceite 10W50 acumuladas en bodega", "SOBRESTOCK"),
    ("Exceso de llantas semislick almacenadas sin venta", "SOBRESTOCK"),
    ("Sobrestock de cascos y accesorios de temporada pasada", "SOBRESTOCK"),
    ("Inventario paralizado de lubricantes sin rotación en 30 días", "SOBRESTOCK"),
    ("Exceso masivo de filtros de aceite antiguos en estantería", "SOBRESTOCK"),

    # Categoría: INCIDENTE_CALIDAD
    ("Batería de gel defectuosa o sulfatada por almacenamiento", "INCIDENTE_CALIDAD"),
    ("Pastillas de freno desgastadas prematuramente por tráfico", "INCIDENTE_CALIDAD"),
    ("Llantas cristalizadas por mal almacenamiento en bodega", "INCIDENTE_CALIDAD"),
    ("Amortiguador monoshock con fuga evidente de aceite hidráulico", "INCIDENTE_CALIDAD"),
    ("Cadena de transmisión oxidada con eslabones gripados peligrosos", "INCIDENTE_CALIDAD"),

    # Categoría: OBSOLESCENCIA
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
            prior = math.log(self.conteo_docs_clase[clase] / self.total_docs)
            likelihood = 0.0
            for token in tokens:
                count = self.conteo_palabras[clase][token]
                prob = (count + 1) / (self.total_palabras_clase[clase] + V)
                likelihood += math.log(prob)
            mejores_score[clase] = prior + likelihood

        return max(mejores_score, key=mejores_score.get)


# 5. ORQUESTACIÓN Y GENERACIÓN DEL REPORTE EXPLICABLE
def ejecutar_sistema_hibrido():
    base_conocimiento = cargar_base_conocimiento()
    experto = SistemaExpertoInventario()
    recuperador = RecuperadorTFIDF(base_conocimiento)
    clasificador = ClasificadorNaiveBayes(DATOS_ENTRENAMIENTO_CLASIFICADOR)

    # Consultas de prueba exigidas por la guía (mínimo 3)
    consultas_prueba = [
        "El kit de arrastre para Dominar 400 está bajo en stock y presenta riesgo de rotura urgente.",
        "Hay un exceso masivo de aceite sintético 10W-50 acumulado en bodega sin rotación en 30 días.",
        "Se detectó un amortiguador monoshock con fuga evidente de aceite hidráulico en el retén principal."
    ]

    resultados_pruebas = []

    print("=" * 80)
    print("SISTEMA HÍBRIDO DE MONITOREO DE INVENTARIO (SEMANA 05)")
    print("=" * 80)

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

    generar_reporte_md(resultados_pruebas, base_conocimiento)


def generar_reporte_md(resultados, base_conocimiento):
    CARPETA_REPORTES.mkdir(parents=True, exist_ok=True)

    lineas = [
        "# Semana 05 - Sistema Híbrido: Sistemas Expertos, Recuperación de Información y Clasificación",
        "## Proyecto: Monitoreo de Inventario Visual y Gestión para Tienda de Motos",
        "",
        "### 1. Descripción del Sistema Híbrido",
        "El sistema inteligente combina tres componentes fundamentales para automatizar y explicar la gestión del inventario de repuestos:",
        "1. **Sistema Experto Basado en Reglas:** Evalúa disparadores críticos de negocio (reorden urgente, alertas de sobrestock, incidentes de calidad).",
        "2. **Recuperador de Información (TF-IDF y Similitud Coseno):** Mapea consultas operativas contra el archivo externo de base de conocimiento persistente (`data/base_conocimiento.txt`).",
        "3. **Clasificador Supervisado (Naive Bayes):** Categoriza automáticamente las solicitudes en etiquetas operativas utilizando más de 20 ejemplos de entrenamiento adaptados al dominio.",
        "",
        "### 2. Base de Conocimiento Integrada (Archivo Externo TXT)",
        f"Se cargaron exitosamente **{len(base_conocimiento)} entradas** especializadas desde `data/base_conocimiento.txt`:",
        ""
    ]

    for entrada in base_conocimiento[:10]:  # Muestra las primeras 10 en el reporte como evidencia resumida
        lineas.append(f"- {entrada}")
    lineas.append(f"- *(... y {len(base_conocimiento) - 10} entradas adicionales cargadas desde el archivo externo)*")

    lineas.extend([
        "",
        "### 3. Pruebas de Ejecución y Resultados Explicables",
        "",
        "| Consulta – Regla – Evidencia – Similitud – Clasificación |",
        "| :--- |"
    ])

    for r in resultados:
        regla_corta = r['regla'].split(":")[0]
        lineas.append(
            f"| **Consulta:** {r['consulta']} <br> **Regla:** `{regla_corta}` <br> **Evidencia:** {r['doc_recuperado']} <br> **Similitud:** `{r['similitud']}` <br> **Clasificación:** **{r['categoria']}** |")

    lineas.extend([
        "",
        "### Detalle Explicativo por Consulta",
        ""
    ])

    for r in resultados:
        lineas.extend([
            f"#### Consulta #{r['num']}: \"{r['consulta']}\"",
            f"- **Regla Activada:** `{r['regla']}`",
            f"- **Información Recuperada:** {r['doc_recuperado']}",
            f"- **Valor de Similitud (TF-IDF):** `{r['similitud']}`",
            f"- **Clasificación Predicha:** `{r['categoria']}`",
            f"- **Trazabilidad Explicable:** El motor evaluó los tokens de la solicitud, activó la directiva experta correspondiente, aisló la norma con mayor proximidad semántica en la base de conocimiento externa y determinó la categoría operativa mediante probabilidad Bayesiana.",
            ""
        ])

    lineas.extend([
        "### 4. Conclusiones y Trazabilidad",
        "La integración del sistema híbrido con archivo de conocimiento independiente permite a la tienda de repuestos de motocicletas automatizar decisiones críticas con total transparencia."
    ])

    RUTA_REPORTE.write_text("\n".join(lineas), encoding="utf-8")
    print(f"\n[OK] Reporte generado exitosamente en: {RUTA_REPORTE}\n")


if __name__ == "__main__":
    ejecutar_sistema_hibrido()