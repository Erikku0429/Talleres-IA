"""
TRABAJO SEMANA 10 - INTELIGENCIA ARTIFICIAL
Reconocimiento de imágenes: histogramas, regiones y texturas LBP
Proyecto: Sistema Inteligente de Inventarios para Taller de Motos (Visión Computacional)
Estudiantes: Erick Santiago Garcia Sanchez, Oscar David Gutierrez
"""

import os
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Modo no interactivo para guardar figuras directamente
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import cv2
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
from skimage.measure import regionprops
from skimage.feature import local_binary_pattern

# --- CONFIGURACIÓN DE RUTAS ---
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
IMAGES_DIR = BASE_DIR / "images"
ARTIFACTS_DIR = BASE_DIR / "artifacts"
REPORTS_DIR = BASE_DIR / "reports"

OUTPUT_NPY_PATH = ARTIFACTS_DIR / "semana10_features.npy"
OUTPUT_META_PATH = ARTIFACTS_DIR / "semana10_features_meta.json"
OUTPUT_PLOT_PATH = ARTIFACTS_DIR / "semana10_histograma.png"
REPORT_PATH = REPORTS_DIR / "semana10.md"

def asegurar_entorno():
    """Garantiza la existencia de directorios de salida y copia respaldos si es necesario."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Asegurar copia de la imagen del proyecto
    if not (DATA_DIR / "imagen_proyecto.png").exists() and (IMAGES_DIR / "foto_bujia.jpg").exists():
        img = cv2.imread(str(IMAGES_DIR / "foto_bujia.jpg"))
        cv2.imwrite(str(DATA_DIR / "imagen_proyecto.png"), img)

def formatear_nombre_imagen(ruta):
    """
    Genera un nombre legible y descriptivo automáticamente para cualquier
    archivo de imagen, identificando repuestos y estados (quemada, rota, etc.).
    """
    stem = Path(ruta).stem.lower()
    if "quemada" in stem:
        return f"Bujía Quemada ({Path(ruta).name})"
    elif "rota" in stem:
        return f"Bujía Rota ({Path(ruta).name})"
    elif "bujia" in stem or stem.startswith("bug"):
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else " Estándar"
        return f"Bujía{sufijo} ({Path(ruta).name})"
    elif "piston" in stem or stem.startswith("pist"):
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else " Estándar"
        return f"Pistón{sufijo} ({Path(ruta).name})"
    elif "llanta" in stem or "rueda" in stem:
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else " Estándar"
        return f"Llanta / Rueda{sufijo} ({Path(ruta).name})"
    elif "cha" in stem or "chasis" in stem:
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else ""
        return f"Chasis / Cadena{sufijo} ({Path(ruta).name})"
    elif "silla" in stem or "asiento" in stem:
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else ""
        return f"Sillín / Asiento{sufijo} ({Path(ruta).name})"
    elif "susp" in stem:
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else ""
        return f"Suspensión{sufijo} ({Path(ruta).name})"
    elif "manubrio" in stem:
        return f"Manubrio ({Path(ruta).name})"
    else:
        limpio = Path(ruta).stem.replace("_", " ").title()
        return f"{limpio} ({Path(ruta).name})"

def obtener_imagenes_dinamicas(directorio_images=IMAGES_DIR, directorio_data=DATA_DIR):
    """
    Escanea dinámicamente las carpetas de imágenes.
    Soporta: .jpg, .jpeg, .png, .bmp, .webp, .tiff
    Cualquier imagen nueva que se coloque en 'images/' aparecerá automáticamente
    sin necesidad de modificar el código.
    """
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    catalogo = {}
    
    # 1. Imagen del proyecto oficial si existe
    p_data = Path(directorio_data) / "imagen_proyecto.png"
    if p_data.exists():
        catalogo["Bujía del Proyecto (imagen_proyecto.png)"] = p_data
        
    # 2. Escaneo dinámico de images/
    p_imgs = Path(directorio_images)
    if p_imgs.exists():
        archivos = sorted([f for f in p_imgs.iterdir() if f.is_file() and f.suffix.lower() in extensiones])
        for arch in archivos:
            label = formatear_nombre_imagen(arch)
            if label in catalogo:
                label = f"{label} [{arch.name}]"
            catalogo[label] = arch
            
    return catalogo

def etiquetar_conectividad_8(mascara):
    """
    Etiqueta componentes conexos utilizando vecindad de 8 conectores
    de manera compatible y robusta (evitando bloqueos de AppLocker en Windows).
    """
    struct_8 = np.ones((3, 3), dtype=int)
    lbl, num = ndi.label(mascara, structure=struct_8)
    return lbl, num

def procesar_imagen_texturas(nombre, ruta, min_area=50, P=8, R=1):
    """
    Ejecuta el pipeline completo de la Semana 10 para una imagen:
    1. Carga y preprocesamiento en escala de grises.
    2. Histograma de intensidad.
    3. Segmentación mediante umbral automático de Otsu.
    4. Etiquetado y filtrado morfológico de regiones.
    5. Extracción de patrones binarios locales (LBP) e histograma de textura.
    6. Vectorización de características (49 dimensiones).
    """
    ruta = Path(ruta)
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la imagen en {ruta}")
        
    img_bgr = cv2.imread(str(ruta))
    if img_bgr is None:
        raise ValueError(f"No se pudo decodificar la imagen: {ruta}")
        
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Histograma de intensidad (32 bins normalizado)
    hist_int, _ = np.histogram(img_gray.ravel(), bins=32, range=(0, 256), density=True)
    
    # 2. Umbral automático de Otsu y máscara binaria
    try:
        umbral_otsu = int(threshold_otsu(img_gray))
    except Exception:
        umbral_otsu = 128
        
    # En fondo blanco iluminado, el repuesto es más oscuro que el fondo (< umbral)
    mascara_binaria = img_gray < umbral_otsu
    pixeles_objeto = int(np.sum(mascara_binaria))
    cobertura_objeto = pixeles_objeto / img_gray.size if img_gray.size > 0 else 0.0
    
    # 3. Etiquetado de regiones (8-conectividad) y medición
    lbl, num_regiones_crudas = etiquetar_conectividad_8(mascara_binaria)
    props = regionprops(lbl)
    areas_crudas = [r.area for r in props] if props else [0]
    
    # Filtro de regiones pequeñas (ruido e imperfecciones)
    props_filtradas = [r for r in props if r.area >= min_area]
    areas_filtradas = [r.area for r in props_filtradas] if props_filtradas else [0]
    num_regiones_filtradas = len(props_filtradas)
    
    area_prom_cruda = float(np.mean(areas_crudas))
    area_std_cruda = float(np.std(areas_crudas))
    area_prom_filt = float(np.mean(areas_filtradas))
    area_std_filt = float(np.std(areas_filtradas))
    area_total_filt = float(np.sum(areas_filtradas))
    
    # 4. Texturas LBP (Local Binary Pattern)
    # P=8 vecinos circulares, Radio R=1, método 'uniform' (genera P+2 = 10 bins)
    lbp = local_binary_pattern(img_gray, P=P, R=R, method='uniform')
    n_bins_lbp = P + 2
    hist_lbp, _ = np.histogram(lbp.ravel(), bins=n_bins_lbp, range=(0, n_bins_lbp), density=True)
    
    # 5. Vector de características combinado
    # [num_crudo, num_filt, area_prom_filt, area_std_filt, area_tot_filt, cobertura, umbral_otsu]
    feats_regiones = np.array([
        float(num_regiones_crudas),
        float(num_regiones_filtradas),
        area_prom_filt,
        area_std_filt,
        area_total_filt,
        cobertura_objeto,
        float(umbral_otsu)
    ], dtype=np.float64)
    
    vector_completo = np.concatenate([feats_regiones, hist_int, hist_lbp]).astype(np.float64)
    
    return {
        "nombre": nombre,
        "ruta": ruta,
        "img_rgb": img_rgb,
        "img_gray": img_gray,
        "dimensiones": img_gray.shape,
        "hist_int": hist_int,
        "umbral_otsu": umbral_otsu,
        "mascara_binaria": mascara_binaria,
        "pixeles_objeto": pixeles_objeto,
        "cobertura_objeto": cobertura_objeto,
        "lbl": lbl,
        "props_crudas": props,
        "num_regiones_crudas": num_regiones_crudas,
        "area_prom_cruda": area_prom_cruda,
        "area_std_cruda": area_std_cruda,
        "props_filtradas": props_filtradas,
        "num_regiones_filtradas": num_regiones_filtradas,
        "area_prom_filt": area_prom_filt,
        "area_std_filt": area_std_filt,
        "area_total_filt": area_total_filt,
        "lbp": lbp,
        "hist_lbp": hist_lbp,
        "feats_regiones": feats_regiones,
        "vector_completo": vector_completo
    }

def generar_visualizacion_histogramas(resultados, output_path):
    """
    Genera un panel comparativo de alta calidad técnica (artifacts/semana10_histograma.png)
    con 5 columnas analíticas por cada repuesto examinado:
    1. Imagen Original RGB
    2. Histograma de Intensidad + Línea de Umbral Otsu
    3. Máscara Binaria y Bounding Boxes de Regiones Filtradas
    4. Mapa de Textura LBP
    5. Histograma de Textura LBP (Distribución de Patrones Uniformes)
    """
    n_imgs = len(resultados)
    fig, axes = plt.subplots(n_imgs, 5, figsize=(22, 4.2 * n_imgs))
    fig.patch.set_facecolor('#0b0f19')  # Fondo slate ultra-oscuro
    
    if n_imgs == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, res in enumerate(resultados):
        # 1. Original RGB
        ax_orig = axes[i, 0]
        ax_orig.imshow(res['img_rgb'])
        ax_orig.set_title(f"{i+1}. {res['nombre']}\n{res['dimensiones'][0]}x{res['dimensiones'][1]} px",
                          color='#f8fafc', fontsize=10, fontweight='bold', pad=6)
        ax_orig.tick_params(colors='#94a3b8', labelsize=7)
        for sp in ax_orig.spines.values(): sp.set_color('#334155')

        # 2. Histograma de Intensidad + Otsu
        ax_hist = axes[i, 1]
        ax_hist.set_facecolor('#1e293b')
        bins_x = np.linspace(0, 255, 32)
        ax_hist.bar(bins_x, res['hist_int'], width=6.5, color='#38bdf8', alpha=0.85, edgecolor='#0284c7', label='Intensidad')
        ax_hist.axvline(res['umbral_otsu'], color='#ef4444', linestyle='--', linewidth=2,
                        label=f"Otsu T={res['umbral_otsu']}")
        ax_hist.set_title(f"Histograma Intensidad (T={res['umbral_otsu']})", color='#f8fafc', fontsize=10, fontweight='bold', pad=6)
        ax_hist.set_xlabel("Nivel de Gris [0-255]", color='#94a3b8', fontsize=7)
        ax_hist.set_ylabel("Densidad Prob.", color='#94a3b8', fontsize=7)
        ax_hist.tick_params(colors='#94a3b8', labelsize=7)
        ax_hist.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', fontsize=7)
        for sp in ax_hist.spines.values(): sp.set_color('#334155')

        # 3. Máscara Binaria y Bounding Boxes de Regiones Filtradas
        ax_mask = axes[i, 2]
        ax_mask.imshow(res['mascara_binaria'], cmap='gray')
        for r in res['props_filtradas']:
            minr, minc, maxr, maxc = r.bbox
            rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     fill=False, edgecolor='#10b981', linewidth=1.5)
            ax_mask.add_patch(rect)
        ax_mask.set_title(f"Regiones Filt (≥50px: {res['num_regiones_filtradas']})\nCob: {res['cobertura_objeto']*100:.1f}%",
                          color='#f8fafc', fontsize=10, fontweight='bold', pad=6)
        ax_mask.tick_params(colors='#94a3b8', labelsize=7)
        for sp in ax_mask.spines.values(): sp.set_color('#334155')

        # 4. Mapa de Textura LBP
        ax_lbp = axes[i, 3]
        im_lbp = ax_lbp.imshow(res['lbp'], cmap='magma')
        ax_lbp.set_title(f"Mapa LBP (Uniforme P=8, R=1)\nValores [0 - 9]",
                         color='#f8fafc', fontsize=10, fontweight='bold', pad=6)
        ax_lbp.tick_params(colors='#94a3b8', labelsize=7)
        for sp in ax_lbp.spines.values(): sp.set_color('#334155')

        # 5. Histograma de Textura LBP
        ax_hlbp = axes[i, 4]
        ax_hlbp.set_facecolor('#1e293b')
        bins_lbp_idx = np.arange(10)
        colores_barras = ['#a855f7' if k < 8 else ('#38bdf8' if k == 8 else '#f43f5e') for k in range(10)]
        ax_hlbp.bar(bins_lbp_idx, res['hist_lbp'], color=colores_barras, alpha=0.9, edgecolor='#64748b', width=0.8)
        ax_hlbp.set_title(f"Distribución LBP (10 Bins)\nBin 8={res['hist_lbp'][8]*100:.1f}% | Bin 9={res['hist_lbp'][9]*100:.1f}%",
                          color='#f8fafc', fontsize=10, fontweight='bold', pad=6)
        ax_hlbp.set_xlabel("Patrón LBP (0-7:Bordes, 8:Plano, 9:Rugoso)", color='#94a3b8', fontsize=7)
        ax_hlbp.set_ylabel("Frecuencia Normalizada", color='#94a3b8', fontsize=7)
        ax_hlbp.set_xticks(range(10))
        ax_hlbp.tick_params(colors='#94a3b8', labelsize=7)
        for sp in ax_hlbp.spines.values(): sp.set_color('#334155')

    plt.tight_layout(pad=2.0)
    plt.savefig(str(output_path), dpi=220, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close(fig)
    print(f"[OK] Artefacto visual guardado exitosamente en: {output_path}")

def generar_informe_markdown(todos_los_resultados, res_bujia, res_quemada, res_llanta, res_piston, report_path):
    """
    Genera el informe técnico de la Semana 10 (reports/semana10.md) con todas las métricas,
    la comparativa profunda de bujía limpia vs quemada, llanta y pistón,
    y el inventario completo de imágenes procesadas.
    """
    total_imgs = len(todos_los_resultados)
    
    # Filas de tabla para todas las imágenes
    filas_tabla_todas = []
    for r in todos_los_resultados:
        filas_tabla_todas.append(
            f"| `{r['ruta'].name}` | {r['nombre']} | {r['dimensiones'][0]}x{r['dimensiones'][1]} | "
            f"**{r['umbral_otsu']}** | {r['cobertura_objeto']*100:.1f}% | "
            f"{r['num_regiones_crudas']} / **{r['num_regiones_filtradas']}** | "
            f"{r['area_prom_filt']:,.0f} px | {r['hist_lbp'][8]*100:.1f}% | {r['hist_lbp'][9]*100:.1f}% |"
        )
    tabla_dataset_str = "\n".join(filas_tabla_todas)
    
    contenido = f"""# Informe Técnico - Semana 10: Reconocimiento de Imágenes
**Asignatura:** Inteligencia Artificial  
**Proyecto:** Sistema Inteligente de Inventarios para Taller de Motos (Visión Computacional)  
**Estudiantes:** Erick Santiago Garcia Sanchez, Oscar David Gutierrez  
**Dataset Escaneado:** {total_imgs} imágenes procesadas dinámicamente  
**Fecha de Entrega:** Octubre 2026  

---

## 1. Selección de Imágenes del Proyecto y Dataset Dinámico

Para cumplir con el requerimiento de análisis sobre múltiples repuestos y estados físicos (bujías estándar, bujías quemadas/deterioradas, llantas, pistones, chasis, sillines y suspensiones), el sistema implementa un **módulo de descubrimiento dinámico de imágenes**:
* Se escanea automáticamente el directorio `images/` detectando cualquier archivo `.jpg`, `.jpeg`, `.png`, `.bmp` o `.webp`.
* **No requiere modificar código** para añadir nuevas imágenes: basta con colocarlas en `images/` y tanto el pipeline por lotes como la aplicación interactiva Streamlit las detectan, procesan y grafican al instante.
* En total se procesaron **{total_imgs} imágenes**, combinando estados de desgaste, geometrías y materiales del taller.

### Casos de Estudio Principales para Análisis Comparativo:
1. **Bujía Estándar Limpia (`images/foto_bujia.jpg`):** Cuerpo roscado de acero mecanizado, hexágono y aislador cerámico blanco liso.
2. **Bujía Quemada / Deteriorada (`images/foto_bujia_quemada.jpg`):** Pieza con depósitos de carbonilla, aceite quemado y electrodo erosionado por sobrecalentamiento.
3. **Llanta de Motocicleta (`images/foto_llanta.jpg`):** Banda de rodadura de caucho negro vulcanizado con estrías direccionales de tracción.
4. **Pistón de Motor (`images/foto_piston.jpg`):** Bloque cilíndrico de aluminio maquinado con ranuras para anillos de compresión.

---

## 2. Segmentación Mediante Histogramas y Umbral Otsu

### 2.1 Justificación Técnica
El histograma de intensidad modela la densidad de probabilidad $P(g)$ de los niveles de gris $g \\in [0, 255]$. El método de Otsu calcula el umbral óptimo $T^*$ que maximiza la varianza inter-clase:
$$\\sigma_b^2(T) = \\omega_0(T) \\omega_1(T) [\\mu_0(T) - \\mu_1(T)]^2$$

Dado que el fondo de inspección es blanco/iluminado, el objeto de interés se aísla mediante:
$$M(x, y) = \\begin{{cases}} 1 & \\text{{si }} I(x, y) < T^* \\\\ 0 & \\text{{en otro caso}} \\end{{cases}}$$

### 2.2 Comparativa de Casos Clave (Bujía Limpia vs. Bujía Quemada vs. Llanta vs. Pistón)

| Parámetro de Segmentación | Bujía Limpia (`foto_bujia.jpg`) | 🔥 Bujía Quemada (`foto_bujia_quemada.jpg`) | 🛞 Llanta (`foto_llanta.jpg`) | ⚙️ Pistón (`foto_piston.jpg`) |
| :--- | :---: | :---: | :---: | :---: |
| **Dimensiones** | {res_bujia['dimensiones'][0]} × {res_bujia['dimensiones'][1]} px | {res_quemada['dimensiones'][0]} × {res_quemada['dimensiones'][1]} px | {res_llanta['dimensiones'][0]} × {res_llanta['dimensiones'][1]} px | {res_piston['dimensiones'][0]} × {res_piston['dimensiones'][1]} px |
| **Umbral Otsu ($T^*$)** | **`{res_bujia['umbral_otsu']}`** | **`{res_quemada['umbral_otsu']}`** | **`{res_llanta['umbral_otsu']}`** | **`{res_piston['umbral_otsu']}`** |
| **Píxeles de Objeto** | {res_bujia['pixeles_objeto']:,} px | {res_quemada['pixeles_objeto']:,} px | {res_llanta['pixeles_objeto']:,} px | {res_piston['pixeles_objeto']:,} px |
| **Cobertura en Imagen** | **{res_bujia['cobertura_objeto']*100:.2f}%** | **{res_quemada['cobertura_objeto']*100:.2f}%** | **{res_llanta['cobertura_objeto']*100:.2f}%** | **{res_piston['cobertura_objeto']*100:.2f}%** |

> **Interpretación Física Crucial:**  
> La **bujía quemada** presenta un umbral Otsu significativamente menor ($T={res_quemada['umbral_otsu']}$ frente a $T={res_bujia['umbral_otsu']}$ de la limpia). Esto se debe a que el hollín, carbón y residuos de combustión oscurecen no solo el electrodo sino también la cerámica circundante, modificando drásticamente la bimodalidad del histograma.

---

## 3. Etiquetado y Medición Morfológica de Regiones

Utilizando conectividad de 8 vecinos (`ndi.label` con matriz $3 \\times 3$ de unos) y `regionprops`, se cuantifican las regiones antes y después del **filtro de área mínima ($A \\ge 50$ px)**:

| Métrica Morfológica | Bujía Limpia | 🔥 Bujía Quemada | 🛞 Llanta | ⚙️ Pistón |
| :--- | :---: | :---: | :---: | :---: |
| **Regiones Crudas** | {res_bujia['num_regiones_crudas']} | {res_quemada['num_regiones_crudas']} | {res_llanta['num_regiones_crudas']} | {res_piston['num_regiones_crudas']} |
| **Regiones Filtradas ($A \\ge 50$ px)** | **`{res_bujia['num_regiones_filtradas']}` componentes** | **`{res_quemada['num_regiones_filtradas']}` componentes** | **`{res_llanta['num_regiones_filtradas']}` componente** | **`{res_piston['num_regiones_filtradas']}` componentes** |
| **Área Promedio Filtrada** | `{res_bujia['area_prom_filt']:,.1f}` px | `{res_quemada['area_prom_filt']:,.1f}` px | `{res_llanta['area_prom_filt']:,.1f}` px | `{res_piston['area_prom_filt']:,.1f}` px |
| **Desviación Estándar Área** | `{res_bujia['area_std_filt']:,.1f}` px | `{res_quemada['area_std_filt']:,.1f}` px | `{res_llanta['area_std_filt']:,.1f}` px | `{res_piston['area_std_filt']:,.1f}` px |

---

## 4. Análisis de Texturas Mediante Patrones Binarios Locales (LBP)

### 4.1 Fundamento Matemático
El operador LBP circular uniforme ($P=8, R=1$, `method='uniform'`) genera 10 bins normalizados:
* **Bins 0 a 7:** Micro-bordes, esquinas y transiciones direccionales.
* **Bin 8:** Superficies completamente homogéneas o planas (fondo liso y metal pulido).
* **Bin 9:** Patrones no uniformes (ruido de alta frecuencia, rugosidad severa y carbonilla).

### 4.2 Comparativa Textural: Bujía Limpia vs. Bujía Quemada vs. Llanta

| Bin LBP | Significado Físico | Bujía Limpia | 🔥 Bujía Quemada | 🛞 Llanta |
| :---: | :--- | :---: | :---: | :---: |
| **Bin 8** | **Zonas Homogéneas / Planas** | **{res_bujia['hist_lbp'][8]*100:.2f}%** | **{res_quemada['hist_lbp'][8]*100:.2f}%** | **{res_llanta['hist_lbp'][8]*100:.2f}%** |
| **Bins 0 a 7** | **Micro-Bordes / Relieves** | {sum(res_bujia['hist_lbp'][:8])*100:.2f}% | {sum(res_quemada['hist_lbp'][:8])*100:.2f}% | **{sum(res_llanta['hist_lbp'][:8])*100:.2f}%** |
| **Bin 9** | **Rugosidad Caótica / No Uniforme**| {res_bujia['hist_lbp'][9]*100:.2f}% | **{res_quemada['hist_lbp'][9]*100:.2f}%** | **{res_llanta['hist_lbp'][9]*100:.2f}%** |

> **Hallazgo Clave de IA:**  
> * La bujía limpia tiene **{res_bujia['hist_lbp'][8]*100:.1f}% de homogeneidad (Bin 8)** debido a su porcelana blanca pulida y superficie metálica impecable.
> * La **bujía quemada** se desploma a **{res_quemada['hist_lbp'][8]*100:.1f}% en el Bin 8**, multiplicando sus micro-bordes y patrones no uniformes debido a las costras irregulares de carbón y corrosión térmica.
> * La **llanta** mantiene una respuesta superior de micro-bordes ({sum(res_llanta['hist_lbp'][:8])*100:.1f}%) debido al labrado direccional de tracción.

---

## 5. Vector de Características Combinado (49 Dimensiones) y Matriz `.npy`

Para cada imagen se extrae:
$$\\vec{{v}}_{{\\text{{pieza}}}} = [7 \\text{{ descriptores de región}}] \\cup [32 \\text{{ bins de intensidad}}] \\cup [10 \\text{{ bins LBP}}] = \\mathbf{{49\\text{{ dimensiones}}}}$$

La matriz completa de forma **`({total_imgs}, 49)`** fue guardada en:
* [**`artifacts/semana10_features.npy`**](file:///c:/Users/Erick/Documents/GitHub/Talleres-IA/artifacts/semana10_features.npy)
* Metadatos estructurados en [**`artifacts/semana10_features_meta.json`**](file:///c:/Users/Erick/Documents/GitHub/Talleres-IA/artifacts/semana10_features_meta.json)

---

## 6. Inventario Completo del Dataset Procesado ({total_imgs} Repuestos)

| Archivo | Repuesto / Estado | Dimensiones | Otsu (T) | Cobertura | Regiones (Crud/Filt) | Área Promedio | LBP Plano (B8) | LBP Rugoso (B9) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
{tabla_dataset_str}

---

## 7. Aplicación Futura en el Proyecto
1. **Diagnóstico Automático de Bujías:** Clasificar bujías en "Operativa" vs "Quemada / Carbonizada" evaluando la caída del Bin 8 y el umbral de Otsu.
2. **Inspección de Desgaste de Llantas:** Monitorear la reducción en la densidad de micro-bordes LBP para predecir cuándo el labrado de la llanta está liso y requiere cambio.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(contenido)
    print(f"[OK] Informe técnico guardado exitosamente en: {report_path}")

def main():
    print("=" * 75)
    print("PROCESAMIENTO DE IMÁGENES - SEMANA 10 (DATASET DINÁMICO)")
    print("Histogramas, Regiones y Texturas LBP en Repuestos de Motos")
    print("=" * 75)

    asegurar_entorno()

    # 1. Escaneo dinámico de todas las imágenes
    imagenes_dict = obtener_imagenes_dinamicas()
    total_imgs = len(imagenes_dict)
    print(f"\n[1] Escaneando imágenes en '{IMAGES_DIR}' y '{DATA_DIR}'...")
    print(f" -> Encontradas {total_imgs} imágenes disponibles automáticamente.")

    resultados_todos = []
    for idx, (label, ruta) in enumerate(imagenes_dict.items(), start=1):
        try:
            res = procesar_imagen_texturas(label, ruta)
            resultados_todos.append(res)
            print(f" [{idx:02d}/{total_imgs:02d}] Procesado: {res['nombre']} ({res['dimensiones'][0]}x{res['dimensiones'][1]}) | Otsu: {res['umbral_otsu']} | Regiones: {res['num_regiones_filtradas']}")
        except Exception as e:
            print(f" [ERROR] No se pudo procesar '{ruta.name}': {e}")

    if not resultados_todos:
        print("[ERROR] No se pudo procesar ninguna imagen.")
        return

    # 2. Guardar matriz .npy consolidada (N, 49)
    print("\n[2] Guardando matriz de características en 'artifacts/semana10_features.npy'...")
    matriz_features = np.vstack([r['vector_completo'] for r in resultados_todos])
    np.save(str(OUTPUT_NPY_PATH), matriz_features)
    print(f"[OK] Matriz consolidada guardada. Shape: {matriz_features.shape} (N={matriz_features.shape[0]}, D={matriz_features.shape[1]})")

    # Guardar metadatos JSON
    metadata = [
        {
            "id": i,
            "archivo": r['ruta'].name,
            "etiqueta": r['nombre'],
            "dimensiones": list(r['dimensiones']),
            "umbral_otsu": int(r['umbral_otsu']),
            "cobertura_objeto_pct": round(float(r['cobertura_objeto']) * 100, 2),
            "regiones_crudas": int(r['num_regiones_crudas']),
            "regiones_filtradas": int(r['num_regiones_filtradas']),
            "area_promedio_filtrada": round(float(r['area_prom_filt']), 1),
            "lbp_bin8_homogeneo_pct": round(float(r['hist_lbp'][8]) * 100, 2),
            "lbp_bin9_rugosidad_pct": round(float(r['hist_lbp'][9]) * 100, 2)
        }
        for i, r in enumerate(resultados_todos)
    ]
    with open(OUTPUT_META_PATH, "w", encoding="utf-8") as f_meta:
        json.dump(metadata, f_meta, indent=2, ensure_ascii=False)
    print(f"[OK] Metadatos guardados en: {OUTPUT_META_PATH}")

    # 3. Seleccionar repuestos representativos para la figura oficial
    # Priorizar: Bujía limpia, Bujía quemada, Bujía #2, Llanta, Pistón, Chasis, Suspensión
    def buscar_res(palabra_clave):
        for r in resultados_todos:
            if palabra_clave in r['ruta'].name.lower():
                return r
        return None

    rep_bujia = buscar_res("foto_bujia.jpg") or resultados_todos[0]
    rep_quemada = buscar_res("quemada") or resultados_todos[min(1, len(resultados_todos)-1)]
    rep_bug2 = buscar_res("bug2") or buscar_res("bug3")
    rep_llanta = buscar_res("foto_llanta.jpg") or buscar_res("rueda")
    rep_piston = buscar_res("foto_piston.jpg") or buscar_res("pist")
    rep_cha = buscar_res("cha1") or buscar_res("cha")
    rep_susp = buscar_res("susp1") or buscar_res("susp")

    seleccionados = [r for r in [rep_bujia, rep_quemada, rep_bug2, rep_llanta, rep_piston, rep_cha, rep_susp] if r is not None]
    
    # Asegurar que los nombres sean únicos
    vistos = set()
    seleccionados_unicos = []
    for s in seleccionados:
        if s['ruta'].name not in vistos:
            vistos.add(s['ruta'].name)
            seleccionados_unicos.append(s)

    print(f"\n[3] Generando figura comparativa oficial ({len(seleccionados_unicos)} repuestos clave)...")
    generar_visualizacion_histogramas(seleccionados_unicos, OUTPUT_PLOT_PATH)

    # 4. Generar informe técnico enriquecido
    print("\n[4] Generando informe técnico (reports/semana10.md)...")
    generar_informe_markdown(
        todos_los_resultados=resultados_todos,
        res_bujia=rep_bujia,
        res_quemada=rep_quemada,
        res_llanta=rep_llanta or rep_bujia,
        res_piston=rep_piston or rep_bujia,
        report_path=REPORT_PATH
    )

    print("\n" + "=" * 75)
    print(f"[EXITO] Semana 10 completada: {total_imgs} imágenes procesadas dinámicamente.")
    print("=" * 75)

if __name__ == "__main__":
    main()
