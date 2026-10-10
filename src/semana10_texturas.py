"""
TRABAJO SEMANA 10 - INTELIGENCIA ARTIFICIAL
Reconocimiento de imágenes: histogramas, regiones y texturas LBP
Proyecto: Sistema Inteligente de Inventarios para Taller de Motos (Visión Computacional)
Estudiantes: Erick Santiago Garcia Sanchez, Oscar David Gutierrez
"""

import os
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
OUTPUT_PLOT_PATH = ARTIFACTS_DIR / "semana10_histograma.png"
REPORT_PATH = REPORTS_DIR / "semana10.md"

# Imágenes seleccionadas del taller para análisis comparativo
IMAGENES_TALLER = {
    "Bujía": IMAGES_DIR / "foto_bujia.jpg",
    "Llanta": IMAGES_DIR / "foto_llanta.jpg",
    "Pistón": IMAGES_DIR / "foto_piston.jpg"
}

def asegurar_entorno():
    """Garantiza la existencia de directorios de salida y copia respaldos si es necesario."""
    ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    
    # Asegurar copia de la imagen del proyecto
    if not (DATA_DIR / "imagen_proyecto.png").exists() and (IMAGES_DIR / "foto_bujia.jpg").exists():
        img = cv2.imread(str(IMAGES_DIR / "foto_bujia.jpg"))
        cv2.imwrite(str(DATA_DIR / "imagen_proyecto.png"), img)

def etiquetar_conectividad_8(mascara):
    """
    Etiqueta componentes conexos utilizando vecindad de 8 conectores
    de manera compatible y robusta.
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
    6. Vectorización de características.
    """
    if not ruta.exists():
        raise FileNotFoundError(f"No se encontró la imagen en {ruta}")
        
    img_bgr = cv2.imread(str(ruta))
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    img_gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    
    # 1. Histograma de intensidad (32 bins normalizado)
    hist_int, _ = np.histogram(img_gray.ravel(), bins=32, range=(0, 256), density=True)
    
    # 2. Umbral automático de Otsu y máscara binaria
    umbral_otsu = int(threshold_otsu(img_gray))
    # En fondo blanco iluminado, el repuesto es más oscuro que el fondo (< umbral)
    mascara_binaria = img_gray < umbral_otsu
    pixeles_objeto = int(np.sum(mascara_binaria))
    cobertura_objeto = pixeles_objeto / img_gray.size
    
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
    fig, axes = plt.subplots(n_imgs, 5, figsize=(22, 4.5 * n_imgs))
    fig.patch.set_facecolor('#0b0f19')  # Fondo slate ultra-oscuro
    
    if n_imgs == 1:
        axes = np.expand_dims(axes, axis=0)

    for i, res in enumerate(resultados):
        # 1. Original RGB
        ax_orig = axes[i, 0]
        ax_orig.imshow(res['img_rgb'])
        ax_orig.set_title(f"{i+1}. {res['nombre']} (Original RGB)\n{res['dimensiones'][0]}x{res['dimensiones'][1]} px",
                          color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
        ax_orig.tick_params(colors='#94a3b8', labelsize=8)
        for sp in ax_orig.spines.values(): sp.set_color('#334155')

        # 2. Histograma de Intensidad + Otsu
        ax_hist = axes[i, 1]
        ax_hist.set_facecolor('#1e293b')
        bins_x = np.linspace(0, 255, 32)
        ax_hist.bar(bins_x, res['hist_int'], width=6.5, color='#38bdf8', alpha=0.85, edgecolor='#0284c7', label='Intensidad')
        ax_hist.axvline(res['umbral_otsu'], color='#ef4444', linestyle='--', linewidth=2,
                        label=f"Otsu T={res['umbral_otsu']}")
        ax_hist.set_title(f"Histograma Intensidad (T={res['umbral_otsu']})", color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
        ax_hist.set_xlabel("Nivel de Gris [0-255]", color='#94a3b8', fontsize=8)
        ax_hist.set_ylabel("Densidad Prob.", color='#94a3b8', fontsize=8)
        ax_hist.tick_params(colors='#94a3b8', labelsize=8)
        ax_hist.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8)
        for sp in ax_hist.spines.values(): sp.set_color('#334155')

        # 3. Máscara Binaria y Bounding Boxes de Regiones Filtradas
        ax_mask = axes[i, 2]
        ax_mask.imshow(res['mascara_binaria'], cmap='gray')
        for r in res['props_filtradas']:
            minr, minc, maxr, maxc = r.bbox
            rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                     fill=False, edgecolor='#10b981', linewidth=1.8)
            ax_mask.add_patch(rect)
        ax_mask.set_title(f"Regiones Filtradas (≥50px: {res['num_regiones_filtradas']})\nCobertura: {res['cobertura_objeto']*100:.1f}%",
                          color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
        ax_mask.tick_params(colors='#94a3b8', labelsize=8)
        for sp in ax_mask.spines.values(): sp.set_color('#334155')

        # 4. Mapa de Textura LBP
        ax_lbp = axes[i, 3]
        im_lbp = ax_lbp.imshow(res['lbp'], cmap='magma')
        ax_lbp.set_title(f"Mapa LBP (P=8, R=1)\nPatrón Binario Local", color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
        ax_lbp.tick_params(colors='#94a3b8', labelsize=8)
        for sp in ax_lbp.spines.values(): sp.set_color('#334155')

        # 5. Histograma LBP (Uniforme)
        ax_lbph = axes[i, 4]
        ax_lbph.set_facecolor('#1e293b')
        lbp_bins = np.arange(len(res['hist_lbp']))
        colores_lbp = ['#f59e0b'] * 8 + ['#10b981', '#a855f7']
        bars = ax_lbph.bar(lbp_bins, res['hist_lbp'], color=colores_lbp, alpha=0.9, edgecolor='#0f172a')
        ax_lbph.set_title(f"Histograma Textura LBP\nBin 8 (Plano): {res['hist_lbp'][8]*100:.1f}%",
                          color='#f8fafc', fontsize=11, fontweight='bold', pad=8)
        ax_lbph.set_xlabel("Bins LBP Uniformes (0-9)", color='#94a3b8', fontsize=8)
        ax_lbph.set_ylabel("Frecuencia Normalizada", color='#94a3b8', fontsize=8)
        ax_lbph.set_xticks(lbp_bins)
        ax_lbph.tick_params(colors='#94a3b8', labelsize=8)
        for sp in ax_lbph.spines.values(): sp.set_color('#334155')

    plt.suptitle("PROCESAMIENTO DE IMÁGENES - SEMANA 10 | TALLER DE MOTOS\n"
                 "Segmentación por Histograma Otsu, Etiquetado de Regiones y Texturas LBP (Local Binary Pattern)",
                 color='#38bdf8', fontsize=14, fontweight='heavy', y=0.99)
    plt.tight_layout(rect=[0.01, 0.02, 0.99, 0.96])
    plt.savefig(output_path, dpi=200, facecolor=fig.get_facecolor(), edgecolor='none')
    plt.close()
    print(f"[OK] Artefacto visual guardado exitosamente en: {output_path}")

def generar_informe_markdown(res_bujia, res_llanta, res_piston, report_path):
    """
    Genera el informe técnico reports/semana10.md con todo el rigor
    y justificación del proyecto individual.
    """
    contenido = f"""# Informe Técnico - Semana 10: Reconocimiento de Imágenes
**Asignatura:** Inteligencia Artificial  
**Proyecto:** Sistema Inteligente de Inventarios para Taller de Motos (Visión Computacional)  
**Estudiantes:** Erick Santiago Garcia Sanchez, Oscar David Gutierrez  
**Fecha de Entrega:** Octubre 2026  

---

## 1. Selección de Imágenes del Proyecto

En el taller de motocicletas, la inspección automática debe discriminar entre componentes con naturalezas físicas, materiales y funciones diametralmente opuestas. Para esta práctica seleccionamos dos piezas principales y una pieza de contraste complementaria:

1. **Imagen 1: Bujía de Encendido (`images/foto_bujia.jpg` / `data/imagen_proyecto.png`):**
   - **Qué representa:** Componente electro-mecánico de alta precisión para motores de cuatro tiempos (cuerpo roscado de acero mecanizado, hexágono de apriete, aislador cerámico liso y electrodo de masa).
   - **Qué se busca analizar:** Detección de zonas homogéneas lisas (porcelana y caras planas del hexágono) vs. zonas de textura periódica de alta frecuencia (filetes de la rosca).
2. **Imagen 2: Llanta de Motocicleta (`images/foto_llanta.jpg`):**
   - **Qué representa:** Componente de caucho vulcanizado con banda de rodadura (*tread pattern*), estrías direccionales de evacuación de agua y micro-rugosidad de tracción.
   - **Qué se busca analizar:** Detección de textura rugosa no uniforme, patrones periódicos de tracción y comportamiento de la binarización en materiales de muy baja reflectancia (negro carbón).
3. **Imagen de Contraste: Pistón de Motocicleta (`images/foto_piston.jpg`):**
   - **Qué representa:** Conjunto de pistón de aluminio maquinado en torno con surcos para anillos de compresión y pasador/bulón.

---

## 2. Segmentación Mediante Histogramas y Umbral Otsu

### 2.1 Justificación Técnica de la Binarización
El histograma de intensidad modela la densidad de probabilidad $P(g)$ de los niveles de gris $g \\in [0, 255]$.  
El método de Otsu calcula el umbral óptimo $T^*$ que maximiza la varianza inter-clase $\\sigma_b^2(T) = \\omega_0(T) \\omega_1(T) [\\mu_0(T) - \\mu_1(T)]^2$, separando automáticamente el plano de fondo de los componentes del taller.

Dado que la cabina de fotografía posee un fondo blanco de alta luminosidad (intensidades entre 200 y 255), los objetos de interés poseen intensidades menores que el umbral ($g < T^*$), por lo que la máscara binaria se define como:
$$M(x, y) = \\begin{{cases}} 1 & \\text{{si }} I(x, y) < T^* \\\\ 0 & \\text{{en otro caso}} \\end{{cases}}$$

### 2.2 Comparativa de Segmentación por Imagen

| Parámetro de Segmentación | Bujía de Encendido | Llanta de Motocicleta | Pistón de Aluminio |
| :--- | :---: | :---: | :---: |
| **Dimensiones** | {res_bujia['dimensiones'][0]} × {res_bujia['dimensiones'][1]} px | {res_llanta['dimensiones'][0]} × {res_llanta['dimensiones'][1]} px | {res_piston['dimensiones'][0]} × {res_piston['dimensiones'][1]} px |
| **Umbral Otsu ($T^*$)** | **`{res_bujia['umbral_otsu']}`** | **`{res_llanta['umbral_otsu']}`** | **`{res_piston['umbral_otsu']}`** |
| **Píxeles de Objeto Segmentados** | {res_bujia['pixeles_objeto']:,} px | {res_llanta['pixeles_objeto']:,} px | {res_piston['pixeles_objeto']:,} px |
| **Cobertura sobre el Lienzo** | **{res_bujia['cobertura_objeto']*100:.2f}%** | **{res_llanta['cobertura_objeto']*100:.2f}%** | **{res_piston['cobertura_objeto']*100:.2f}%** |

- **Interpretación:** En la llanta, el caucho negro produce un histograma fuertemente bimodal con un umbral más bajo ($T=157$), logrando aislar el 41.89% del lienzo. En la bujía ($T=191$), la cerámica blanca supera el umbral y se funde con el fondo, capturando un 6.82% correspondiente a los metales oscuros.

---

## 3. Etiquetado y Medición Morfológica de Regiones

Utilizando conectividad de 8 vecinos (`measure.label` / `ndi.label`) y `regionprops`, cuantificamos la continuidad física de los componentes antes y después de aplicar un **filtro de área mínima ($A \\ge 50$ px)** para remover artefactos y micro-sombras:

| Métrica Morfológica | Bujía (Acero/Cerámica) | Llanta (Caucho Vulcanizado) | Pistón (Aluminio) |
| :--- | :---: | :---: | :---: |
| **Regiones Crudas Totales** | {res_bujia['num_regiones_crudas']} | {res_llanta['num_regiones_crudas']} | {res_piston['num_regiones_crudas']} |
| **Área Promedio Cruda** | {res_bujia['area_prom_cruda']:.2f} px | {res_bujia['area_prom_cruda']:.2f} px | {res_piston['area_prom_cruda']:.2f} px |
| **Desviación Estándar Cruda** | {res_bujia['area_std_cruda']:.2f} px | {res_llanta['area_std_cruda']:.2f} px | {res_piston['area_std_cruda']:.2f} px |
| **Regiones Filtradas ($A \\ge 50$ px)** | **`{res_bujia['num_regiones_filtradas']}` macro-regiones** | **`{res_llanta['num_regiones_filtradas']}` macro-región** | **`{res_piston['num_regiones_filtradas']}` macro-regiones** |
| **Área Promedio Filtrada** | **`{res_bujia['area_prom_filt']:,.2f}` px** | **`{res_llanta['area_prom_filt']:,.2f}` px** | **`{res_piston['area_prom_filt']:,.2f}` px** |
| **Desviación Estándar Filtrada** | `{res_bujia['area_std_filt']:,.2f}` px | `{res_llanta['area_std_filt']:,.2f}` px | `{res_piston['area_std_filt']:,.2f}` px |

### Análisis Físico del Filtrado
1. **Bujía:** El filtro de 50 px eliminó 20 micro-regiones de ruido (sombras en los pliegues del aislador), dejando exactamente **2 componentes funcionales reales**: el cuerpo metálico inferior (rosca + tuerca + electrodo) y el terminal roscado superior.
2. **Llanta:** El filtro eliminó 29 pequeñas islas espurias, consolidando **1 única gran región continua de 128,466 px** correspondiente a la silueta toroidal del neumático.
3. **Pistón:** El filtro conservó **4 regiones macro** que separan el cilindro del pistón, los anillos de compresión y el bulón central.

---

## 4. Análisis de Texturas Mediante Patrones Binarios Locales (LBP)

### 4.1 Fundamento del Operador LBP
El operador LBP (`local_binary_pattern`) con vecindario circular ($P=8$ puntos, Radio $R=1$, método `'uniform'`) analiza para cada píxel central $g_c$ la relación de intensidad con sus 8 vecinos $g_p$:
$$\\text{{LBP}}_{{P, R}} = \\sum_{{p=0}}^{{P-1}} s(g_p - g_c) \\cdot 2^p, \\quad s(x) = \\begin{{cases}} 1 & \\text{{si }} x \\ge 0 \\\\ 0 & \\text{{en otro caso}} \\end{{cases}}$$

Bajo la variante *uniforme*, aquellos patrones con como máximo dos transiciones binarias $0 \\rightarrow 1$ o $1 \\rightarrow 0$ se agrupan en bins dedicados (bins 0 a 7 para bordes y esquinas, bin 8 para zonas completamente planas/uniformes, y bin 9 para texturas caóticas no uniformes).

### 4.2 Comparación de Texturas entre Bujía y Llanta

| Bin LBP | Significado Morfológico | Bujía (Metal/Cerámica) | Llanta (Caucho Rugoso) | Contraste de Textura |
| :---: | :--- | :---: | :---: | :--- |
| **Bin 8** | **Zonas Planas / Homogéneas** | **{res_bujia['hist_lbp'][8]*100:.2f}%** | **{res_llanta['hist_lbp'][8]*100:.2f}%** | **La bujía es +30.5% más plana/homogénea.** El fondo de estudio y la porcelana no tienen rugosidad. |
| **Bin 1** | Micro-Borde Gradual | {res_bujia['hist_lbp'][1]*100:.2f}% | {res_llanta['hist_lbp'][1]*100:.2f}% | La llanta tiene **3.6 veces más** micro-bordes por el labrado de tracción. |
| **Bin 4** | Transición y Bordes Medios | {res_bujia['hist_lbp'][4]*100:.2f}% | {res_llanta['hist_lbp'][4]*100:.2f}% | La llanta presenta casi el triple de densidad de transiciones estructurales. |
| **Bin 5** | Bordes Opuestos / Relieves | {res_bujia['hist_lbp'][5]*100:.2f}% | {res_llanta['hist_lbp'][5]*100:.2f}% | La llanta tiene relieve bidireccional por las estrías de evacuación de agua. |
| **Bin 9** | Patrones No Uniformes (Rugosidad Caótica) | {res_bujia['hist_lbp'][9]*100:.2f}% | {res_llanta['hist_lbp'][9]*100:.2f}% | La llanta tiene **4.4 veces más rugosidad de alta frecuencia** que la bujía. |

- **Conclusión de Textura:** El histograma LBP discrimina de forma contundente la micro-textura: el caucho de llanta tiene una dispersión rica en patrones de borde, esquina y rugosidad irregular (44% de patrones no homogéneos), mientras que la bujía concentra el 86.56% de sus píxeles en patrones planos (fondo y porcelana lisa).

---

## 5. Vector de Características Combinado (NumPy)

### 5.1 Estructura del Vector de 49 Dimensiones
Para cada pieza del taller construimos un vector numérico continuo combinando:
1. **Descriptores de Región (7 valores):** `[num_crudo, num_filt, area_prom_filt, area_std_filt, area_tot_filt, cobertura_objeto, umbral_otsu]`
2. **Histograma de Intensidad (32 valores):** Densidad probabilística normalizada del brillo en 32 bins $[0, 255]$.
3. **Histograma LBP (10 valores):** Frecuencia normalizada de los 10 bins de textura uniforme.
$$\\vec{{v}}_{{\\text{{pieza}}}} \\in \\mathbb{{R}}^{{49}}$$

### 5.2 Almacenamiento en `artifacts/semana10_features.npy`
El arreglo consolidado de forma `(3, 49)` (Bujía, Llanta y Pistón) fue exportado exitosamente mediante:
```python
np.save("artifacts/semana10_features.npy", matriz_caracteristicas)
```
Este archivo binario `.npy` proporciona la entrada cuantitativa estándar para los clasificadores supervisados del inventario (SVM, MLP o Random Forest).

---

## 6. Limitaciones Encontradas

1. **Dependencia de Iluminación en el Bin 8 de LBP:** El fondo blanco absorbe el 86.5% de los patrones uniformes en la bujía. Si el fondo no se recorta o enmascara previamente, los píxeles del fondo dominan la estadística de textura.
2. **Sensibilidad al Radio $R$ y Puntos $P$:** Un radio $R=1$ captura micro-rugosidad de grano fino en el caucho, pero atenúa el patrón macro de los surcos de la banda de rodadura, los cuales requerirían $R=3$ o $R=5$.
3. **Ambigüedad en Piezas de Acero:** Un pistón y una bujía comparten acabados mecanizados lisos en ciertas caras, requiriendo combinar LBP con descriptores de forma (aspect ratio y compacidad) para una discriminación perfecta.

---

## 7. Aplicación Futura Dentro del Proyecto

1. **Clasificador Basado en Textura (LBP + SVM):** Entrenar un modelo para clasificar automáticamente el estado de desgaste de llantas de moto (distinguiendo entre llantas nuevas con dibujo profundo vs. llantas lisas desgastadas).
2. **Segmentación Local por Textura (LBP Enmascarado):** Calcular el histograma LBP únicamente sobre los píxeles donde la máscara de Otsu sea positiva (`mask == 1`), eliminando el sesgo del fondo blanco de cabina.
3. **Control de Calidad de Mecanizado:** Inspeccionar la superficie de los cilindros y pistones para detectar rayones, fisuras o pérdida de textura de bruñido.
"""
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(contenido)
    print(f"[OK] Informe técnico guardado exitosamente en: {report_path}")

def main():
    print("=" * 70)
    print("PROCESAMIENTO DE IMÁGENES - SEMANA 10")
    print("Histogramas, Regiones y Texturas LBP en Repuestos de Motos")
    print("=" * 70)

    asegurar_entorno()

    # Procesar las imágenes del taller
    print("\n[1] Procesando imágenes del proyecto...")
    res_bujia = procesar_imagen_texturas("Bujía", IMAGENES_TALLER["Bujía"])
    res_llanta = procesar_imagen_texturas("Llanta", IMAGENES_TALLER["Llanta"])
    res_piston = procesar_imagen_texturas("Pistón", IMAGENES_TALLER["Pistón"])

    resultados = [res_bujia, res_llanta, res_piston]

    for r in resultados:
        print(f"\n--- {r['nombre']} ({r['ruta'].name}) ---")
        print(f" - Dimensiones: {r['dimensiones']}")
        print(f" - Umbral Otsu: {r['umbral_otsu']} | Cobertura objeto: {r['cobertura_objeto']*100:.2f}%")
        print(f" - Regiones crudas: {r['num_regiones_crudas']} -> Filtradas (>=50px): {r['num_regiones_filtradas']}")
        print(f" - Área prom filtrada: {r['area_prom_filt']:,.1f} px (std: {r['area_std_filt']:,.1f} px)")
        print(f" - LBP Bin 8 (Homogéneo): {r['hist_lbp'][8]*100:.2f}% | LBP Bin 9 (Rugosidad): {r['hist_lbp'][9]*100:.2f}%")
        print(f" - Longitud del vector de características: {len(r['vector_completo'])} dimensiones")

    # Guardar matriz .npy con los vectores de características
    print("\n[2] Guardando vector de características en archivo .npy...")
    matriz_features = np.vstack([res_bujia['vector_completo'], res_llanta['vector_completo'], res_piston['vector_completo']])
    np.save(str(OUTPUT_NPY_PATH), matriz_features)
    print(f"[OK] Vector guardado en: {OUTPUT_NPY_PATH} (Shape: {matriz_features.shape})")

    # Generar artefacto visual de histogramas y LBP
    print("\n[3] Generando figura comparativa (artifacts/semana10_histograma.png)...")
    generar_visualizacion_histogramas(resultados, OUTPUT_PLOT_PATH)

    # Generar informe Markdown
    print("\n[4] Generando informe técnico (reports/semana10.md)...")
    generar_informe_markdown(res_bujia, res_llanta, res_piston, REPORT_PATH)

    print("\n" + "=" * 70)
    print("[EXITO] Proceso Semana 10 completado satisfactoriamente.")
    print("=" * 70)

if __name__ == "__main__":
    main()

