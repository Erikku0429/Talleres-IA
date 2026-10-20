"""
TRABAJO SEMANA 11 - INTELIGENCIA ARTIFICIAL
Benchmark de modelos y decisión técnica con evidencia
Proyecto: Sistema Inteligente de Inventarios para Taller de Motos (Visión Computacional)
Modalidad: Individual
Estudiante: Erick Santiago Garcia Sanchez / Oscar David Gutierrez
"""

import sys
import time
from pathlib import Path
import json
import numpy as np
import pandas as pd
import cv2
import warnings
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

from skimage.filters import threshold_otsu
from skimage.measure import regionprops
from skimage.feature import local_binary_pattern
import scipy.ndimage as ndi

from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, classification_report

# --- CONFIGURACIÓN DE RUTAS Y TERMINAL ---
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

BENCHMARK_CSV_PATH = ARTIFACTS_DIR / "benchmark.csv"
DATASET_CACHE_PATH = ARTIFACTS_DIR / "semana11_dataset.npz"

def asegurar_entorno():
    """Crea los directorios necesarios para artefactos y reportes."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

def clasificar_archivo_repuesto(nombre_archivo: str) -> str:
    """
    Identifica la clase canónica de repuesto de motocicleta a partir del nombre
    del archivo de imagen, garantizando correspondencia con el catálogo del taller.
    """
    stem = Path(nombre_archivo).stem.lower()
    if "bujia" in stem or stem.startswith("bug") or "proyecto" in stem:
        return "Bujia"
    elif "piston" in stem or stem.startswith("pist"):
        return "Piston"
    elif "llanta" in stem or "rueda" in stem:
        return "Llanta"
    elif "cha" in stem or "chasis" in stem:
        return "Chasis"
    elif "silla" in stem or "asiento" in stem:
        return "Sillin"
    elif "susp" in stem:
        return "Suspension"
    return None

def extraer_vector_49d(img_bgr: np.ndarray, min_area: int = 50, P: int = 8, R: int = 1) -> np.ndarray:
    """
    Extrae el vector de 49 características del pipeline de Visión Computacional (Semana 10):
    - 7 descriptores morfométricos de región con segmentación Otsu y 8-conectividad.
    - 32 bins de histograma de intensidades en escala de grises.
    - 10 bins de textura LBP (Local Binary Pattern) uniforme (P=8, R=1).
    """
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Histograma de intensidades (32 bins normalizado)
    hist_int, _ = np.histogram(img_gray.ravel(), bins=32, range=(0, 256), density=True)
    
    # 2. Umbral de Otsu y máscara de objeto
    try:
        t_otsu = int(threshold_otsu(img_gray))
    except Exception:
        t_otsu = 128
        
    mask = img_gray < t_otsu
    struct_8 = np.ones((3, 3), dtype=int)
    lbl, num_crudas = ndi.label(mask, structure=struct_8)
    props = regionprops(lbl)
    
    props_filt = [r for r in props if r.area >= min_area]
    areas_filt = [r.area for r in props_filt] if props_filt else [0]
    num_filt = len(props_filt)
    area_prom = float(np.mean(areas_filt))
    area_std = float(np.std(areas_filt))
    area_tot = float(np.sum(areas_filt))
    cobertura = float(np.sum(mask)) / img_gray.size if img_gray.size > 0 else 0.0
    
    # 3. Textura LBP (10 bins)
    lbp = local_binary_pattern(img_gray, P=P, R=R, method='uniform')
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=P + 2, range=(0, P + 2), density=True)
    
    feats_reg = np.array([
        float(num_crudas), float(num_filt), area_prom,
        area_std, area_tot, cobertura, float(t_otsu)
    ], dtype=np.float64)
    
    return np.concatenate([feats_reg, hist_int, hist_lbp]).astype(np.float64)

def construir_o_cargar_dataset(forzar_reconstruccion: bool = False):
    """
    Carga el dataset de características del proyecto del taller de motos.
    Si existe el archivo en caché (.npz), lo carga en milisegundos.
    Si no existe o se fuerza reconstrucción, procesa las imágenes reales de images/
    aplicando transformaciones físicas realistas de la estación de inspección:
    - Imagen original
    - Inversión horizontal (simetría de ensamble de piezas)
    - Variación lumínica alta (+20% brillo, luz cenital)
    - Variación lumínica baja (-20% brillo, sombra de operario)
    - Rotación a 90 grados
    - Rotación a 180 grados
    """
    if DATASET_CACHE_PATH.exists() and not forzar_reconstruccion:
        datos = np.load(DATASET_CACHE_PATH, allow_pickle=True)
        X = datos["X"]
        y = datos["y"]
        clases = datos["clases"].tolist()
        print(f"[DATASET] Cargado desde caché: {len(X)} muestras, {X.shape[1]} características.")
        return X, y, clases

    print("[DATASET] Construyendo dataset a partir de imágenes reales en 'images/'...")
    archivos = []
    if IMAGES_DIR.exists():
        archivos.extend(sorted([f for f in IMAGES_DIR.iterdir() if f.suffix.lower() in {'.jpg', '.jpeg', '.png'}]))
    
    p_proj = DATA_DIR / "imagen_proyecto.png"
    if p_proj.exists() and p_proj not in archivos:
        archivos.append(p_proj)

    X_list = []
    y_list = []

    for f in archivos:
        clase = clasificar_archivo_repuesto(f.name)
        if clase is None:
            continue  # Pieza fuera de dominio / trampa (ej. manubrio)
            
        img = cv2.imread(str(f))
        if img is None:
            continue

        # Generar variaciones fotométricas y geométricas realistas del taller
        variaciones = [
            img,
            cv2.flip(img, 1),
            cv2.convertScaleAbs(img, alpha=1.2, beta=15),
            cv2.convertScaleAbs(img, alpha=0.8, beta=-15),
            cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE),
            cv2.rotate(img, cv2.ROTATE_180),
        ]

        for v in variaciones:
            feat_vec = extraer_vector_49d(v)
            X_list.append(feat_vec)
            y_list.append(clase)

    X = np.array(X_list, dtype=np.float64)
    y = np.array(y_list)
    clases = sorted(list(set(y)))

    np.savez_compressed(DATASET_CACHE_PATH, X=X, y=y, clases=np.array(clases))
    print(f"[DATASET] Construido y guardado en caché: {len(X)} muestras, {X.shape[1]} dimensiones, {len(clases)} clases.")
    return X, y, clases

def ejecutar_benchmark(num_corridas: int = 3, test_size: float = 0.25, seed_base: int = 42):
    """
    Ejecuta el protocolo de benchmarking comparativo:
    1. Mismo conjunto de datos y misma partición (train_test_split estratificado).
    2. Tres modelos: Logistic Regression, Random Forest, MLPClassifier.
    3. Medición precisa de Accuracy y tiempo de entrenamiento (fit_seconds).
    4. Mínimo 3 ejecuciones para cuantificar la variabilidad del hardware / CPU.
    5. Exportación a artifacts/benchmark.csv con evidencia verificable.
    """
    asegurar_entorno()
    X, y, clases = construir_o_cargar_dataset()

    print("\n" + "=" * 72)
    print("  BENCHMARK DE MODELOS - SEMANA 11: PROYECTO TALLER DE MOTOS")
    print(f"  Total Muestras: {len(X)} | Características: {X.shape[1]} | Clases: {len(clases)}")
    print(f"  Clases evaluadas: {', '.join(clases)}")
    print("=" * 72)

    # Partición única y estricta para garantizar una comparación equitativa
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=seed_base, stratify=y
    )
    print(f"[SPLIT] Partición fijada (Semilla {seed_base}): {len(X_train)} entrenamiento, {len(X_test)} prueba.")

    # Definición de los tres modelos requeridos con configuraciones técnicas sólidas
    modelos_fabrica = {
        "logistic": lambda: make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=1500, random_state=seed_base)
        ),
        "forest": lambda: RandomForestClassifier(
            n_estimators=250,
            random_state=seed_base,
            n_jobs=-1
        ),
        "mlp": lambda: make_pipeline(
            StandardScaler(),
            MLPClassifier(hidden_layer_sizes=(64,), max_iter=450, random_state=seed_base)
        ),
    }

    # Registro de métricas a través de múltiples corridas
    resultados_por_corrida = {nombre: {"accuracy": [], "fit_seconds": []} for nombre in modelos_fabrica}

    print(f"\n[CORRIDAS] Ejecutando {num_corridas} iteraciones bajo idéntico split para medir variabilidad del hardware...")
    for corrida in range(1, num_corridas + 1):
        print(f"\n--- Corrida {corrida}/{num_corridas} ---")
        for nombre, constructor in modelos_fabrica.items():
            modelo = constructor()
            
            # Medición exacta de tiempo de ajuste
            t_inicio = time.perf_counter()
            modelo.fit(X_train, y_train)
            t_entrenamiento = time.perf_counter() - t_inicio
            
            # Evaluación de exactitud sobre el conjunto de prueba
            y_pred = modelo.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            
            resultados_por_corrida[nombre]["accuracy"].append(acc)
            resultados_por_corrida[nombre]["fit_seconds"].append(t_entrenamiento)
            
            print(f"  [{nombre:8s}] -> Accuracy: {acc*100:6.2f}% | Tiempo: {t_entrenamiento:7.4f} seg")

    # Consolidación de estadísticas
    filas_resumen = []
    for nombre in modelos_fabrica:
        accs = resultados_por_corrida[nombre]["accuracy"]
        tiempos = resultados_por_corrida[nombre]["fit_seconds"]
        
        # El accuracy se mantiene idéntico si la semilla se conserva, tomamos el valor nominal
        acc_promedio = float(np.mean(accs))
        tiempo_promedio = float(np.mean(tiempos))
        tiempo_std = float(np.std(tiempos))
        
        filas_resumen.append({
            "modelo": nombre,
            "accuracy": round(acc_promedio, 4),
            "fit_seconds": round(tiempo_promedio, 4),
            "fit_seconds_std": round(tiempo_std, 4),
            "corridas": num_corridas
        })

    # Creación del DataFrame de Pandas ordenado por Accuracy descendente
    df_completo = pd.DataFrame(filas_resumen).sort_values("accuracy", ascending=False)
    
    # DataFrame oficial exacto para artifacts/benchmark.csv (modelo, accuracy, fit_seconds)
    df_benchmark_csv = df_completo[["modelo", "accuracy", "fit_seconds"]].copy()

    print("\n" + "=" * 72)
    print("  TABLA FINAL DEL BENCHMARK (Ordenada por Accuracy):")
    print(df_completo[["modelo", "accuracy", "fit_seconds", "fit_seconds_std"]].to_string(index=False))
    print("=" * 72)

    # Guardar archivo oficial
    df_benchmark_csv.to_csv(BENCHMARK_CSV_PATH, index=False)
    print(f"\n[OK] Archivo oficial de benchmark guardado exitosamente en:")
    print(f"     -> {BENCHMARK_CSV_PATH}")

    # Análisis de diferencias de rendimiento
    mejor_modelo = df_completo.iloc[0]["modelo"]
    mejor_acc = df_completo.iloc[0]["accuracy"]
    mas_rapido = df_completo.sort_values("fit_seconds").iloc[0]["modelo"]
    menor_tiempo = df_completo.sort_values("fit_seconds").iloc[0]["fit_seconds"]

    print("\n[ANALISIS] Síntesis técnica de comparación:")
    print(f" • Mayor Accuracy:   '{mejor_modelo}' con {mejor_acc*100:.2f}% de acierto.")
    print(f" • Menor Tiempo:     '{mas_rapido}' entrenando en {menor_tiempo:.4f} segundos.")
    
    acc_log = df_completo.loc[df_completo['modelo'] == 'logistic', 'accuracy'].values[0]
    acc_rf = df_completo.loc[df_completo['modelo'] == 'forest', 'accuracy'].values[0]
    t_log = df_completo.loc[df_completo['modelo'] == 'logistic', 'fit_seconds'].values[0]
    t_rf = df_completo.loc[df_completo['modelo'] == 'forest', 'fit_seconds'].values[0]
    
    dif_acc = (acc_rf - acc_log) * 100
    ratio_t = t_rf / t_log if t_log > 0 else 1.0
    print(f" • Brecha RF vs Log: Random Forest supera a Logistic en {dif_acc:+.2f}% de accuracy,")
    print(f"                     pero requiere {ratio_t:.1f}x más tiempo de cómputo.")

    # --- ANÁLISIS COMPLEMENTARIO DE VARIACIÓN DE PARTICIÓN (SEMILLAS) ---
    print("\n" + "-" * 72)
    print("  ANÁLISIS DE ROBUSTEZ: VARIACIÓN ESTADÍSTICA EN 3 PARTICIONES DISTINTAS")
    print("-" * 72)
    semillas_test = [42, 100, 2026]
    res_semillas = {m: [] for m in modelos_fabrica}
    for s in semillas_test:
        X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_size, random_state=s, stratify=y)
        for nom in modelos_fabrica:
            if nom == "logistic":
                clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=s))
            elif nom == "forest":
                clf = RandomForestClassifier(n_estimators=250, random_state=s, n_jobs=-1)
            else:
                clf = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=450, random_state=s))
            clf.fit(X_tr, y_tr)
            res_semillas[nom].append(accuracy_score(y_te, clf.predict(X_te)))

    filas_semillas = []
    for nom, accs in res_semillas.items():
        filas_semillas.append({
            "modelo": nom,
            "acc_semilla_42": round(accs[0], 4),
            "acc_semilla_100": round(accs[1], 4),
            "acc_semilla_2026": round(accs[2], 4),
            "acc_promedio": round(float(np.mean(accs)), 4),
            "acc_std": round(float(np.std(accs)), 4)
        })
    df_semillas = pd.DataFrame(filas_semillas).sort_values("acc_promedio", ascending=False)
    print(df_semillas.to_string(index=False))
    print("=" * 72 + "\n")

    return df_completo

if __name__ == "__main__":
    reconstruir = "--rebuild" in sys.argv
    ejecutar_benchmark(num_corridas=3, test_size=0.25, seed_base=42)

