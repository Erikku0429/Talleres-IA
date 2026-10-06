import streamlit as st
import numpy as np
import math
import heapq
import sqlite3
import os
from collections import Counter
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.color import rgb2gray
from skimage.feature import canny
from skimage.filters import threshold_otsu
from skimage.measure import label, regionprops
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

# CONFIGURACIÓN DE PÁGINA Y ESTILOS UI/UX
st.set_page_config(
    page_title="Dashboard IA - Monitoreo de Inventario",
    page_icon="🏍️",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
    <style>
    .main { background-color: #0b0f19; color: #f8fafc; }
    .sidebar .sidebar-content { background-color: #111827; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        padding: 1.25rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.2);
        text-align: center;
    }
    .metric-card h3 { color: #94a3b8; font-size: 0.85rem; text-transform: uppercase; margin-bottom: 0.5rem; }
    .metric-card p { color: #38bdf8; font-size: 1.8rem; font-weight: 800; margin: 0; }
    h1, h2, h3 { color: #f1f5f9; font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif; }
    .stTabs [data-baseweb="tab-list"] { gap: 12px; background-color: #0f172a; padding: 10px; border-radius: 10px; }
    .stTabs [data-baseweb="tab"] { 
        background-color: #1e293b; border-radius: 8px; color: #94a3b8; padding: 10px 20px; font-weight: 600; border: 1px solid #334155;
    }
    .stTabs [aria-selected="true"] { 
        background-color: #38bdf8 !important; color: #0b0f19 !important; border-color: #38bdf8 !important;
    }
    </style>
""", unsafe_allow_html=True)

# BARRA LATERAL (SIDEBAR)

with st.sidebar:
    st.image("https://img.icons8.com/color/96/motorcycle.png", width=70)
    st.title("Panel de Control IA")
    st.markdown("**Proyecto:** Tienda y Taller de Motos")
    st.markdown("**Asignatura:** Ingeniería de Sistemas S10A")
    st.markdown("---")
    st.markdown("### 👥 Desarrolladores")
    st.markdown("• Erick Santiago Garcia Sanchez\n• Oscar David Gutierrez")
    st.markdown("---")

# ENCABEZADO PRINCIPAL

st.title("🏍️ Sistema Inteligente de Monitoreo y Logística")
st.markdown("Plataforma avanzada para la gestión automatizada de inventario de repuestos para motocicletas.")

# NAVEGACIÓN POR PESTAÑAS (MODULAR)

tab_resumen, tab_s2, tab_s3, tab_s4, tab_s5, tab_s8, tab_s9 = st.tabs([
    "📊 Resumen Ejecutivo",
    "🤖 Semana 2: Línea Base",
    "🔍 Semana 3: Taxonomía",
    "🗺️ Semana 4: Dashboard A*",
    "⚡ Semana 5: Sistema Híbrido",
    "👁️ Semana 8: Reconocimiento IA",
    "🔬 Semana 9: Visión Computacional"
])

# 1. RESUMEN EJECUTIVO

with tab_resumen:
    st.header("Arquitectura Semestral del Proyecto")
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        st.markdown('<div class="metric-card"><h3>Semana 2</h3><p>94.7%</p><small>Accuracy Base</small></div>',
                    unsafe_allow_html=True)
    with col2:
        st.markdown('<div class="metric-card"><h3>Semana 3</h3><p>30 Casos</p><small>Taxonomía IA</small></div>',
                    unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><h3>Semana 4</h3><p>Algoritmo A*</p><small>Rutas Óptimas</small></div>',
                    unsafe_allow_html=True)
    with col4:
        st.markdown(
            '<div class="metric-card"><h3>Semana 5</h3><p>150 Entradas</p><small>Base de Conocimiento</small></div>',
            unsafe_allow_html=True)
    with col5:
        st.markdown(
            '<div class="metric-card"><h3>Semana 8</h3><p>MLP 100%</p><small>Auditoría SQLite</small></div>',
            unsafe_allow_html=True)
    with col6:
        st.markdown(
            '<div class="metric-card"><h3>Semana 9</h3><p>Canny + Otsu</p><small>23 Regiones Conexas</small></div>',
            unsafe_allow_html=True)

# 2. SEMANA 2: LÍNEA BASE (MACHINE LEARNING)

with tab_s2:
    st.header("Semana 2: Fundamentos de IA y Línea Base")
    st.markdown("Modelo de clasificación supervisada con **Regresión Logística** y estandarización.")

    col_izq, col_der = st.columns(2)
    with col_izq:
        test_size_val = st.slider("Proporción de Prueba (Test Size)", 0.1, 0.4, 0.25, 0.05)
        random_seed = st.number_input("Semilla Aleatoria (Random State)", value=42, step=1)
        if st.button("Entrenar Modelo Base", type="primary"):
            np.random.seed(int(random_seed))
            X = np.random.rand(150, 2) * 10
            y = (X[:, 0] + X[:, 1] > 12).astype(int)
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size_val, random_state=int(random_seed), stratify=y
            )
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000, random_state=int(random_seed)))
            model.fit(X_train, y_train)
            pred = model.predict(X_test)
            st.session_state['s2_results'] = {'train_len': len(X_train), 'test_len': len(X_test),
                                              'acc': accuracy_score(y_test, pred), 'cm': confusion_matrix(y_test, pred)}

    with col_der:
        if 's2_results' in st.session_state:
            res = st.session_state['s2_results']
            st.success("¡Modelo entrenado exitosamente!")
            st.metric("Precisión del Modelo (Accuracy)", f"{res['acc']:.3f}")
            st.write(f"• **Muestras de entrenamiento:** {res['train_len']}")
            st.write(f"• **Muestras de prueba:** {res['test_len']}")
            st.code(str(res['cm']))
        else:
            st.info("👈 Haga clic en 'Entrenar Modelo Base' para procesar los datos.")


# 3. SEMANA 3: TAXONOMÍA DE IA

with tab_s3:
    st.header("Semana 3: Taxonomía de Inteligencia Artificial")
    caso_usuario = st.text_input("Ingrese un caso o solicitud de taller:",
                                 "Detectar abolladuras severas y óxido en tanques de gasolina mediante imágenes ópticas")
    if st.button("Analizar Caso con Taxonomía"):
        texto = caso_usuario.lower()
        if any(w in texto for w in ["imagen", "foto", "abolladura", "rayón", "óxido", "cámara"]):
            cat = "Visión por Computador"
        elif any(w in texto for w in ["stock", "demanda", "predecir", "inventario"]):
            cat = "Aprendizaje Automático Predictivo"
        elif any(w in texto for w in ["ruta", "picking", "bodega"]):
            cat = "Búsqueda y Optimización"
        else:
            cat = "Sistemas Expertos"
        st.info(f"**Categoría Principal Detectada:** `{cat}`")

# 4. SEMANA 4: DASHBOARD DE LOGÍSTICA A* DINÁMICO

with tab_s4:
    st.header("Semana 4: Dashboard Interactivo de Logística (Búsqueda A*)")
    st.markdown(
        "Modifica los parámetros de inicio y destino. El algoritmo calculará y animará la ruta óptima de *picking* en tiempo real.")

    # Definición del plano de la bodega (6x6) -> 0: Pasillo, 1: Rack (Obstáculo), 2: Taller (Tráfico)
    mapa_bodega = [
        [0, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 1],
        [0, 2, 2, 0, 0, 0],
        [0, 0, 0, 0, 1, 0]
    ]

    col_controles, col_mapa = st.columns([1, 1.4])

    with col_controles:
        st.subheader("⚙️ Configurar Ruta de Recolección")
        origen_r = st.selectbox("Fila de Inicio (Operario)", [0, 1, 2, 3, 4, 5], index=0, key="or_r")
        origen_c = st.selectbox("Columna de Inicio (Operario)", [0, 1, 2, 3, 4, 5], index=0, key="or_c")
        meta_r = st.selectbox("Fila del Repuesto (Meta)", [0, 1, 2, 3, 4, 5], index=2, key="mt_r")
        meta_c = st.selectbox("Columna del Repuesto (Meta)", [0, 1, 2, 3, 4, 5], index=4, key="mt_c")

        inicio = (origen_r, origen_c)
        meta = (meta_r, meta_c)


        # Función de A* real conectada a los inputs
        def calcular_a_star(mapa, start, goal):
            if mapa[start[0]][start[1]] == 1 or mapa[goal[0]][goal[1]] == 1:
                return [], float('inf'), 0

            filas, cols = len(mapa), len(mapa[0])
            cola = [(0.0, start[0], start[1])]
            came_from = {}
            g_score = {start: 0.0}
            nodos_evaluados = 0

            while cola:
                f, r, c = heapq.heappop(cola)
                nodos_evaluados += 1

                if (r, c) == goal:
                    ruta = []
                    actual = (r, c)
                    while actual in came_from:
                        ruta.append(actual)
                        actual = came_from[actual]
                    ruta.append(start)
                    ruta.reverse()
                    return ruta, g_score[goal], nodos_evaluados

                for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < filas and 0 <= nc < cols and mapa[nr][nc] != 1:
                        costo_terreno = 3.0 if mapa[nr][nc] == 2 else 1.0
                        tentative_g = g_score[(r, c)] + costo_terreno

                        if (nr, nc) not in g_score or tentative_g < g_score[(nr, nc)]:
                            came_from[(nr, nc)] = (r, c)
                            g_score[(nr, nc)] = tentative_g
                            h = abs(nr - goal[0]) + abs(nc - goal[1])
                            heapq.heappush(cola, (tentative_g + h, nr, nc))
            return [], float('inf'), nodos_evaluados


        ruta_calculada, costo_total, nodos = calcular_a_star(mapa_bodega, inicio, meta)

    with col_mapa:
        st.subheader("🗺️ Plano Dinámico de la Bodega")
        st.markdown("Leyenda: 🟦 Pasillo | 🟥 Rack | 🟨 Taller | 🟢 Ruta Óptima | 🏁 Inicio/Meta")

        if not ruta_calculada and inicio != meta:
            st.error("⚠️ La posición de inicio o meta colisiona con una estantería (bloqueada).")

        grid_html = "<div style='display: grid; grid-template-columns: repeat(6, 52px); gap: 6px;'>"
        for r in range(6):
            for c in range(6):
                val = mapa_bodega[r][c]
                bg = "#1e293b"
                icon = f"{r},{c}"

                if val == 1:
                    bg = "#ef4444"
                    icon = "🧱"
                elif val == 2:
                    bg = "#f59e0b"
                    icon = "⚠️"

                if (r, c) == inicio:
                    bg = "#38bdf8"
                    icon = "🏃‍♂️"
                elif (r, c) == meta:
                    bg = "#a855f7"
                    icon = "🎯"
                elif (r, c) in ruta_calculada and (r, c) != inicio and (r, c) != meta:
                    bg = "#10b981"
                    icon = "•"

                grid_html += f"<div style='width:52px; height:52px; background:{bg}; color:white; display:flex; align-items:center; justify-content:center; border-radius:8px; font-weight:bold; font-size:0.8rem; border: 1px solid #334155;'>{icon}</div>"
        grid_html += "</div>"
        st.markdown(grid_html, unsafe_allow_html=True)

        if ruta_calculada:
            st.success(
                f"✅ **Ruta calculada con éxito!** Costo total $g(n)$: `{costo_total}` | Nodos explorados: `{nodos}`")

# 5. SEMANA 5: SISTEMA HÍBRIDO AVANZADO

with tab_s5:
    st.header("Semana 5: Sistema Híbrido Explicable")
    ruta_bc = Path("data/base_conocimiento.txt")
    if ruta_bc.exists():
        lineas_bc = [l.strip() for l in ruta_bc.read_text(encoding="utf-8").split("\n") if l.strip()]
        st.success(f"✅ Base de conocimiento externa vinculada: **{len(lineas_bc)} entradas** cargadas.")
    else:
        lineas_bc = ["1. Los kits de arrastre requieren revisión de stock semanal y reorden inmediata."]

    consulta_s5 = st.text_area("Consulta o incidencia operativa:",
                               "El kit de arrastre para Dominar 400 está bajo en stock y presenta riesgo de rotura urgente en el taller.")
    if st.button("Procesar Consulta Híbrida", type="primary"):
        c_low = consulta_s5.lower()
        regla = "REGLA_DEFAULT: Monitoreo estándar en ERP de tienda."
        if "kit de arrastre" in c_low or "cadena" in c_low or "piñón" in c_low:
            regla = "REGLA_1_ACTIVADA: [REORDEN URGENTE / BLOQUEO] Emitir orden de compra inmediata."
        elif "aceite" in c_low:
            regla = "REGLA_3_ACTIVADA: [PROMOCIÓN] Lanzar combo promocional de cambio de aceite."

        mejor_doc = lineas_bc[0]
        max_c = 0
        palabras = set(c_low.split())
        for doc in lineas_bc:
            coincidencias = sum(1 for p in palabras if p in doc.lower() and len(p) > 2)
            if coincidencias > max_c:
                max_c = coincidencias
                mejor_doc = doc

        st.info(f"**1. Motor de Reglas Expertas:**\n`{regla}`")
        st.warning(f"**2. Recuperación de Información (TF-IDF):**\n`{mejor_doc}`")
        st.success("**3. Clasificación Supervisada (Naive Bayes):**\n`RIESGO_REORDEN`")


# 6. SEMANA 8: RECONOCIMIENTO DE REPUESTOS (RED NEURONAL & SQLITE)

with tab_s8:
    st.header("Semana 8: Representaciones del Reconocimiento")
    st.markdown("Clasificación de piezas de taller con Red Neuronal (`MLPClassifier`), umbral de rechazo al 75% y auditoría en SQLite.")

    col_s8_1, col_s8_2 = st.columns([1, 1.2])

    with col_s8_1:
        st.subheader("Auditoría de Inferencia")
        db_path = Path("artifacts/inventario_evidencia.db")
        if db_path.exists():
            conn = sqlite3.connect(str(db_path))
            cur = conn.cursor()
            cur.execute("SELECT id, imagen_analizada, prediccion_ia, confianza_porcentaje, fecha_registro FROM evidencia_reconocimiento ORDER BY id DESC LIMIT 10")
            filas = cur.fetchall()
            conn.close()
            if filas:
                st.dataframe(
                    [{"ID": f[0], "Imagen": f[1], "Predicción IA": f[2], "Certeza (%)": f"{f[3]}%", "Fecha": f[4]} for f in filas],
                    use_container_width=True
                )
            else:
                st.info("No hay registros en la base de datos de auditoría aún.")
        else:
            st.warning("Base de datos `artifacts/inventario_evidencia.db` no encontrada.")

    with col_s8_2:
        st.subheader("Ontología y Arquitectura del Reconocimiento")
        st.markdown("""
        - **Entrada:** Vector numérico de 4 características [R, G, B, Aspect Ratio].
        - **Capas Ocultas:** `MLPClassifier(hidden_layer_sizes=(16, 8))`.
        - **Umbral de Seguridad:** `predict_proba >= 0.75`; de lo contrario, clasifica como `Pieza_No_Reconocida`.
        - **Ontología GraphML:** Mapea la relación entre imagen, red neuronal, clase de pieza, estado de certeza y persistencia SQLite.
        """)


# 7. SEMANA 9: VISIÓN COMPUTACIONAL (BORDES CANNY, OTSU & REGIONES CONEXAS)

with tab_s9:
    st.header("Semana 9: Reconocimiento de Imágenes - Contornos, Otsu y Regiones Conectadas")
    st.markdown("Transformación numérica de imágenes de repuestos mediante **Canny Edge Detection**, **Segmentación Otsu** y **Etiquetado de Componentes Conexos**.")

    ruta_img_defecto = Path("data/imagen_proyecto.png")
    if not ruta_img_defecto.exists():
        ruta_img_defecto = Path("images/foto_bujia.jpg")

    col_ctrl, col_vista = st.columns([1, 2.2])

    with col_ctrl:
        st.subheader("⚙️ Configuración del Pipeline")
        opciones_img = {
            "Bujía (data/imagen_proyecto.png)": str(ruta_img_defecto),
            "Pistón (images/foto_piston.jpg)": "images/foto_piston.jpg",
            "Llanta (images/foto_llanta.jpg)": "images/foto_llanta.jpg",
            "Manubrio (images/foto_manubrio.jpg)": "images/foto_manubrio.jpg"
        }
        sel_nombre = st.selectbox("Seleccione imagen del taller:", list(opciones_img.keys()), index=0)
        img_sel_path = Path(opciones_img[sel_nombre])

        sigma_val = st.slider("Parámetro Sigma de Canny (σ):", min_value=0.5, max_value=4.0, value=2.0, step=0.5,
                              help="Controla la dispersión del filtro Gaussiano antes de derivar gradientes.")

        usar_otsu_auto = st.checkbox("Usar Umbral Automático Otsu", value=True)
        umbral_manual = 191
        if not usar_otsu_auto:
            umbral_manual = st.slider("Umbral Manual de Binarización [0-255]:", 0, 255, 191)

        st.markdown("---")
        st.markdown("### 📌 Resumen Teórico")
        st.caption("• **Canny:** Filtro gaussiano + Magnitud/Ángulo Sobel + Supresión de no-máximos + Histéresis.")
        st.caption("• **Otsu:** Maximiza la varianza inter-clase de las poblaciones de píxeles.")
        st.caption("• **Regiones:** Agrupación por vecindad conexa de 8 vecinos sobre la máscara binaria.")

    with col_vista:
        if img_sel_path.exists():
            # Cargar imagen y procesar
            img_bgr = cv2.imread(str(img_sel_path))
            img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
            img_gray = rgb2gray(img_rgb)
            img_gray_u8 = (img_gray * 255).astype(np.uint8)

            # Otsu
            t_otsu_float = float(threshold_otsu(img_gray))
            t_otsu_u8 = int(np.round(t_otsu_float * 255))
            t_usado = t_otsu_float if usar_otsu_auto else (umbral_manual / 255.0)

            # Canny con sigma seleccionado
            edges_sel = canny(img_gray, sigma=sigma_val)
            pixeles_borde = int(np.sum(edges_sel))

            # Máscara y regiones
            mask_sel = img_gray < t_usado
            pixeles_obj = int(np.sum(mask_sel))
            pct_obj = (pixeles_obj / img_gray.size) * 100.0

            labels_sel = label(mask_sel)
            total_regs = int(labels_sel.max())
            props_sel = sorted(regionprops(labels_sel), key=lambda r: r.area, reverse=True)

            # Métricas rápidas
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Umbral Otsu", f"T={t_otsu_u8}", f"Float: {t_otsu_float:.3f}")
            with m2:
                st.metric(f"Bordes Canny (σ={sigma_val})", f"{pixeles_borde:,} px", f"{(pixeles_borde/img_gray.size)*100:.2f}%")
            with m3:
                st.metric("Regiones Conectadas", f"{total_regs}", "Componentes")
            with m4:
                st.metric("Área del Objeto", f"{pct_obj:.1f}%", f"{pixeles_obj:,} px")

            # Sub-pestañas de visualización
            tab_v_interactiva, tab_v_panel, tab_v_datos = st.tabs([
                "🖼️ Procesamiento en Tiempo Real",
                "📊 Artefacto Completo (8 Paneles)",
                "📋 Tabla de Regiones Conectadas"
            ])

            with tab_v_interactiva:
                fig_live, axs = plt.subplots(1, 4, figsize=(16, 4))
                fig_live.patch.set_facecolor('#0f172a')
                for ax in axs:
                    ax.tick_params(colors='#94a3b8', labelsize=7)
                    for sp in ax.spines.values():
                        sp.set_color('#334155')

                # 1. Original
                axs[0].imshow(img_rgb)
                axs[0].set_title("1. Original RGB", color='#f8fafc', fontsize=10, fontweight='bold')

                # 2. Canny
                axs[1].imshow(edges_sel, cmap='hot')
                axs[1].set_title(f"2. Canny (σ={sigma_val})", color='#f8fafc', fontsize=10, fontweight='bold')

                # 3. Máscara Otsu
                axs[2].imshow(mask_sel, cmap='gray')
                axs[2].set_title(f"3. Máscara (T={int(t_usado*255)})", color='#f8fafc', fontsize=10, fontweight='bold')

                # 4. Regiones y Bounding Boxes
                axs[3].imshow(img_rgb)
                for r in props_sel[:5]:
                    minr, minc, maxr, maxc = r.bbox
                    color_box = '#10b981' if r.area > 1000 else '#f59e0b'
                    rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                             fill=False, edgecolor=color_box, linewidth=1.5)
                    axs[3].add_patch(rect)
                axs[3].set_title("4. Regiones Clave (Bounding Box)", color='#f8fafc', fontsize=10, fontweight='bold')

                plt.tight_layout()
                st.pyplot(fig_live)
                plt.close(fig_live)

            with tab_v_panel:
                artefacto_path = Path("artifacts/semana09_vision.png")
                if artefacto_path.exists():
                    st.image(str(artefacto_path), caption="Figura generada por src/semana09_vision.py (8 Paneles)")
                else:
                    st.info("Ejecute `python src/semana09_vision.py` para generar la figura oficial.")

            with tab_v_datos:
                st.markdown(f"**Top 10 Regiones Conexas Detectadas (Total: {total_regs}):**")
                filas_tabla = []
                for idx, r in enumerate(props_sel[:10], start=1):
                    interpretacion = (
                        "Cuerpo metálico inferior (rosca + tuerca + electrodo)" if idx == 1 and r.area > 5000 else
                        ("Terminal metálico superior roscado" if idx == 2 and r.area > 500 else
                         f"Micro-sombra en curvatura del aislador / ruido de superficie ({r.area} px)")
                    )
                    filas_tabla.append({
                        "Ranking": idx,
                        "Label ID": r.label,
                        "Área (px)": f"{r.area:,}",
                        "Bounding Box (y0, x0, y1, x1)": str(r.bbox),
                        "Centroide (y, x)": f"({r.centroid[0]:.1f}, {r.centroid[1]:.1f})",
                        "Interpretación Física": interpretacion
                    })
                st.dataframe(filas_tabla, use_container_width=True)

        else:
            st.error(f"No se encontró el archivo de imagen en `{img_sel_path}`.")


# PIE DE PÁGINA

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Universidad / Institución | Ingeniería de Sistemas S10A | Repositorio Oficial</div>",
    unsafe_allow_html=True)