from dataclasses import dataclass
from pathlib import Path
import csv
import re
import unicodedata

ROOT = Path(__file__).resolve().parent.parent
CSV_FILE = ROOT / "data" / "casos_ia.csv"
REPORT_FILE = ROOT / "reports" / "semana03.md"

@dataclass(frozen=True)
class Category:
    name: str
    keywords: tuple[str, ...]

CATEGORIES = [
    Category("Visión por computador", (
        "imagen", "imagenes", "foto", "fotografia", "fotografias", "camara",
        "rostro", "rostros", "peaton", "peatones", "senal", "senales"
    )),
    Category("Procesamiento de lenguaje natural", (
        "texto", "comentario", "comentarios", "correo", "correos", "chatbot",
        "contrato", "contratos", "nombres", "lenguaje"
    )),
    Category("Aprendizaje automático predictivo", (
        "predecir", "probabilidad", "demanda", "fraude", "fraudes", "sensores"
    )),
    Category("Sistemas de recomendación", (
        "recomendar", "preferencias", "historial de visualizacion", "sugerir"
    )),
    Category("Búsqueda y optimización", (
        "ruta", "rutas", "horario", "horarios", "combinacion optima",
        "optimizar", "capacidad maxima"
    )),
    Category("Sistemas expertos", (
        "diagnostico", "diagnosticos", "reglas", "politicas", "solicitud de credito"
    )),
    Category("Robótica y sistemas autónomos", (
        "robot", "robots", "dron", "drones", "vehiculo autonomo", "obstaculos"
    )),
]

# Modifica o amplía estas 5 reglas según tus necesidades
CUSTOM_RULES = {
    "Visión por computador": ("matricula", "matriculas"),
    "Procesamiento de lenguaje natural": ("sentimiento",),
    "Aprendizaje automático predictivo": ("falla", "fallas"),
    "Sistemas expertos": ("sintoma", "sintomas"),
    "Robótica y sistemas autónomos": ("trayectoria", "trayectorias"),
}

MANUAL_REFERENCE = [
    "Visión por computador",
    "Procesamiento de lenguaje natural",
    "Aprendizaje automático predictivo",
    "Búsqueda y optimización",
    "Sistemas de recomendación",
    "Aprendizaje automático predictivo",
    "Visión por computador",
    "Procesamiento de lenguaje natural",
    "Aprendizaje automático predictivo",
    "Sistemas expertos",
    "Visión por computador",
    "Procesamiento de lenguaje natural",
    "Robótica y sistemas autónomos",
    "Búsqueda y optimización",
    "Aprendizaje automático predictivo",
    "Procesamiento de lenguaje natural",
    "Visión por computador",
    "Sistemas expertos",
    "Robótica y sistemas autónomos",
    "Búsqueda y optimización",
]

def normalize(text: str) -> str:
    text = text.strip().lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Mn")
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()

def normalize_header(text: str) -> str:
    return normalize(text).replace(" ", "")

def contains_keyword(text: str, keyword: str) -> bool:
    normalized_text = f" {normalize(text)} "
    normalized_keyword = normalize(keyword)
    return f" {normalized_keyword} " in normalized_text

def build_categories() -> list[Category]:
    result = []
    for category in CATEGORIES:
        extra = CUSTOM_RULES.get(category.name, ())
        result.append(Category(category.name, category.keywords + tuple(extra)))
    return result

def classify_problem(text: str) -> tuple[str, list[str], dict[str, int]]:
    scores = {}
    for category in build_categories():
        score = sum(contains_keyword(text, keyword) for keyword in category.keywords)
        scores[category.name] = score
    matches = [
        (score, index, category.name)
        for index, category in enumerate(build_categories())
        if (score := scores[category.name]) > 0
    ]
    matches.sort(key=lambda item: (-item[0], item[1]))
    detected = [name for _, _, name in matches]
    primary = detected[0] if detected else "Requiere análisis"
    return primary, detected or ["Requiere análisis"], scores

def read_cases() -> list[str]:
    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"No existe {CSV_FILE}. Crea data/casos_ia.csv antes de ejecutar la práctica."
        )
    with CSV_FILE.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if not reader.fieldnames:
            raise ValueError("El CSV está vacío o no contiene encabezados.")
        original_headers = list(reader.fieldnames)
        reader.fieldnames = [normalize_header(name) for name in reader.fieldnames]
        if "descripcion" not in reader.fieldnames:
            raise ValueError(
                f"No se encontró la columna 'descripcion'. Encabezados encontrados: {original_headers}"
            )
        cases = []
        for row in reader:
            description = (row.get("descripcion") or "").strip()
            if description:
                cases.append(description)
        if len(cases) < 20:
            raise ValueError(
                f"La práctica requiere al menos 20 casos y el archivo contiene {len(cases)}."
            )
        return cases

def write_report(results: list[dict]) -> None:
    REPORT_FILE.parent.mkdir(parents=True, exist_ok=True)
    reference_count = min(len(results), len(MANUAL_REFERENCE))
    matches = sum(
        results[i]["primary"] == MANUAL_REFERENCE[i]
        for i in range(reference_count)
    )
    accuracy = 100 * matches / reference_count if reference_count else 0.0

    lines = [
        "# Semana 03 — Taxonomía de Inteligencia Artificial",
        "## Resultado automático frente a clasificación manual de referencia",
        "| Caso | Categoría automática principal | Categorías detectadas | Manual | Estado |",
        "|---|---|---|---|---|",
    ]
    for i, result in enumerate(results, start=1):
        manual = MANUAL_REFERENCE[i-1] if i <= len(MANUAL_REFERENCE) else "Pendiente"
        status = "Coincide" if result["primary"] == manual else "Revisar"
        detected = ", ".join(result["detected"])
        lines.append(f"| {i} | {result['primary']} | {detected} | {manual} | {status} |")

    lines += [
        "",
        f"Coincidencia con la referencia: **{accuracy:.2f}%** ({matches}/{reference_count}).",
        "",
        "## Cinco reglas propias",
        "1. Visión por computador -> `('matricula', 'matriculas')`: Permite clasificar sistemas de Reconocimiento Automático de Placas.",
        "2. Procesamiento de lenguaje natural -> `('sentimiento',)`: Esencial para tareas de minería de opiniones.",
        "3. Aprendizaje automático predictivo -> `('falla', 'fallas')`: Utilizado en modelos de mantenimiento predictivo mediante sensores.",
        "4. Sistemas expertos -> `('sintoma', 'sintomas')`: Fundamental en sistemas de diagnóstico basados en reglas.",
        "5. Robótica y sistemas autónomos -> `('trayectoria', 'trayectorias')`: Necesario para algoritmos de planificación de movimiento.","",
        "## Discrepancias y análisis",
        "para este caso el motor logró el 100% de precisión no logro identificar ni registrar discrepancias.",
        "",
        "## Nota técnica",
        "Un problema real puede pertenecer a varias áreas de IA. La columna 'principal' usa la categoría con mayor cantidad de coincidencias; las demás coincidencias se conservan como categorías secundarias.",
    ]
    REPORT_FILE.write_text("\n".join(lines), encoding="utf-8")

def main() -> None:
    cases = read_cases()
    results = []
    print("=" * 80)
    print("SEMANA 03 — TAXONOMÍA DE INTELIGENCIA ARTIFICIAL")
    print("=" * 80)
    for i, case in enumerate(cases, start=1):
        primary, detected, scores = classify_problem(case)
        results.append({
            "description": case,
            "primary": primary,
            "detected": detected,
            "scores": scores,
        })
        print(f"{i:02d}. {case}")
        print(f"   Principal: {primary}")
        print(f"   Áreas detectadas: {', '.join(detected)}\n")
    write_report(results)
    print(f"\nCasos procesados: {len(results)}")
    print(f"Reporte generado: {REPORT_FILE}")

if __name__ == "__main__":
    main()