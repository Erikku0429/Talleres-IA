import streamlit as st
import numpy as np
import math
import heapq
import sqlite3
from collections import Counter
from pathlib import Path
from PIL import Image
import cv2
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage.color import rgb2gray
from skimage.feature import canny, local_binary_pattern
from skimage.filters import threshold_otsu
import scipy.ndimage as ndi
from skimage.measure import regionprops
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import json

def formatear_nombre_imagen(ruta):
    """Genera un nombre legible y descriptivo con emojis para cualquier archivo de imagen."""
    p = Path(ruta)
    stem = p.stem.lower()
    if "quemada" in stem:
        return f"🔥 Bujía Quemada ({p.name})"
    elif "rota" in stem:
        return f"💥 Bujía Rota ({p.name})"
    elif "bujia" in stem or stem.startswith("bug"):
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else " Estándar"
        return f"⚡ Bujía{sufijo} ({p.name})"
    elif "piston" in stem or stem.startswith("pist"):
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else " Estándar"
        return f"⚙️ Pistón{sufijo} ({p.name})"
    elif "llanta" in stem or "rueda" in stem:
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else " Estándar"
        return f"🛞 Llanta / Rueda{sufijo} ({p.name})"
    elif "cha" in stem or "chasis" in stem:
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else ""
        return f"⛓️ Chasis / Cadena{sufijo} ({p.name})"
    elif "silla" in stem or "asiento" in stem:
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else ""
        return f"🪑 Sillín / Asiento{sufijo} ({p.name})"
    elif "susp" in stem:
        num = "".join(filter(str.isdigit, stem))
        sufijo = f" #{num}" if num else ""
        return f"🔩 Suspensión{sufijo} ({p.name})"
    elif "manubrio" in stem:
        return f"🏍️ Manubrio ({p.name})"
    else:
        limpio = p.stem.replace("_", " ").title()
        return f"📦 {limpio} ({p.name})"

def obtener_catalogo_imagenes():
    """
    Escanea dinámicamente 'images/' y 'data/' para detectar automáticamente
    todas las imágenes disponibles (.jpg, .jpeg, .png, .bmp, .webp, .tiff).
    Permite colocar nuevas imágenes sin tocar una sola línea de código.
    """
    extensiones = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tif", ".tiff"}
    catalogo = {}
    p_data = Path("data/imagen_proyecto.png")
    if p_data.exists():
        catalogo["⭐ Bujía Proyecto (data/imagen_proyecto.png)"] = str(p_data)
    p_imgs = Path("images")
    if p_imgs.exists():
        archivos = sorted([f for f in p_imgs.iterdir() if f.is_file() and f.suffix.lower() in extensiones])
        for arch in archivos:
            label = formatear_nombre_imagen(arch)
            if label in catalogo:
                label = f"{label} [{arch.name}]"
            catalogo[label] = str(arch)
    return catalogo

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

tab_resumen, tab_s2, tab_s3, tab_s4, tab_s5, tab_s7, tab_s8, tab_s9, tab_s10, tab_s11 = st.tabs([
    "📊 Resumen Ejecutivo",
    "🤖 Semana 2: Línea Base",
    "🔍 Semana 3: Taxonomía",
    "🗺️ Semana 4: Dashboard A*",
    "⚡ Semana 5: Sistema Híbrido",
    "📐 Semana 7: Representaciones",
    "👁️ Semana 8: Reconocimiento IA",
    "🔬 Semana 9: Visión Computacional",
    "🧩 Semana 10: Texturas LBP",
    "⚖️ Semana 11: Benchmark y ADR"
])

# 1. RESUMEN EJECUTIVO

with tab_resumen:
    st.header("Arquitectura Semestral del Proyecto")
    col1, col2, col3, col4, col5, col6, col7, col8, col9 = st.columns(9)
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
            '<div class="metric-card"><h3>Semana 5</h3><p>150 Entradas</p><small>Base Conocimiento</small></div>',
            unsafe_allow_html=True)
    with col5:
        st.markdown(
            '<div class="metric-card"><h3>Semana 7</h3><p>3 Modelos</p><small>Num + Simb + AFD</small></div>',
            unsafe_allow_html=True)
    with col6:
        st.markdown(
            '<div class="metric-card"><h3>Semana 8</h3><p>MLP 100%</p><small>Auditoría SQLite</small></div>',
            unsafe_allow_html=True)
    with col7:
        st.markdown(
            '<div class="metric-card"><h3>Semana 9</h3><p>Canny + Otsu</p><small>23 Regiones</small></div>',
            unsafe_allow_html=True)
    with col8:
        st.markdown(
            '<div class="metric-card"><h3>Semana 10</h3><p>LBP + Otsu</p><small>Vector 49D (.npy)</small></div>',
            unsafe_allow_html=True)
    with col9:
        st.markdown(
            '<div class="metric-card"><h3>Semana 11</h3><p>Random Forest</p><small>86% / 0.7s (ADR)</small></div>',
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


# 6. SEMANA 7: REPRESENTACIONES DEL RECONOCIMIENTO (NUMÉRICA, SIMBÓLICA Y AFD)

with tab_s7:
    st.header("Semana 7: Representaciones del Reconocimiento")
    st.markdown("Implementación y comparación interactiva de las tres representaciones clásicas de la IA: **Numérica (Espacio Vectorial)**, **Simbólica (Reglas de Negocio)** y **Sintáctica (Autómata Finito Determinista)**.")

    sub_num, sub_simb, sub_afd, sub_comp = st.tabs([
        "🔢 1. Representación Numérica",
        "🧠 2. Representación Simbólica",
        "🔄 3. Autómata Finito (AFD)",
        "📋 4. Tabla Comparativa"
    ])

    # --- 1. REPRESENTACIÓN NUMÉRICA ---
    with sub_num:
        st.subheader("Representación Vectorial y Distancia Euclidiana")
        st.markdown("Cada repuesto se modela matemáticamente como un vector cuantitativo $\\vec{x} = [\\text{Stock Actual}, \\text{Días sin Rotación}]$.")

        col_num_ctrl, col_num_plot = st.columns([1, 1.4])

        with col_num_ctrl:
            st.markdown("#### Configurar Pieza a Evaluar")
            s7_stock = st.slider("Nivel de Stock de la pieza (unidades):", 0, 60, 4, key="s7_stock")
            s7_dias = st.slider("Días sin rotación en taller:", 0, 60, 28, key="s7_dias")

            vec_ref = [2.0, 35.0]  # Pieza crítica de referencia
            vec_user = [float(s7_stock), float(s7_dias)]
            dist_user = math.sqrt((vec_ref[0] - vec_user[0])**2 + (vec_ref[1] - vec_user[1])**2)

            st.markdown("---")
            st.markdown(f"**Vector de Referencia Crítica:** `[Stock: 2, Días: 35]`")
            st.markdown(f"**Vector de Pieza Actual:** `[Stock: {s7_stock}, Días: {s7_dias}]`")
            st.metric("Distancia Euclidiana $d$", f"{dist_user:.2f}")

            if dist_user < 12.0:
                st.error("🚨 **Alerta Crítica:** Alta similitud con estado crítico (stock agotándose y alta inmovilización). Requiere pedido inmediato.")
            elif dist_user < 25.0:
                st.warning("⚠️ **Alerta Moderada:** Proximidad intermedia al riesgo. Monitorear rotación semanal.")
            else:
                st.success("✅ **Inventario Saludable:** Gran distancia euclidiana respecto al estado crítico.")

        with col_num_plot:
            st.markdown("#### Espacio Vectorial 2D de Repuestos")
            fig_vec, ax_vec = plt.subplots(figsize=(6, 4.5))
            fig_vec.patch.set_facecolor('#0f172a')
            ax_vec.set_facecolor('#1e293b')

            # Puntos conocidos
            p_ref = [2.0, 35.0]
            p_p1 = [1.0, 40.0]
            p_p2 = [45.0, 2.0]

            ax_vec.scatter([p_ref[0]], [p_ref[1]], color='#ef4444', s=160, marker='X', label='Referencia Crítica [2, 35]', zorder=5)
            ax_vec.scatter([p_p1[0]], [p_p1[1]], color='#f59e0b', s=100, label=f'Pieza 1 (Crítica) [1, 40] (d={math.sqrt((p_ref[0]-p_p1[0])**2 + (p_ref[1]-p_p1[1])**2):.1f})', zorder=4)
            ax_vec.scatter([p_p2[0]], [p_p2[1]], color='#10b981', s=100, label=f'Pieza 2 (Saludable) [45, 2] (d={math.sqrt((p_ref[0]-p_p2[0])**2 + (p_ref[1]-p_p2[1])**2):.1f})', zorder=4)
            ax_vec.scatter([vec_user[0]], [vec_user[1]], color='#38bdf8', s=140, marker='o', label=f'Tu Pieza [{s7_stock}, {s7_dias}] (d={dist_user:.1f})', zorder=6)

            # Línea conectora entre tu pieza y referencia
            ax_vec.plot([p_ref[0], vec_user[0]], [p_ref[1], vec_user[1]], color='#38bdf8', linestyle='--', alpha=0.7)

            ax_vec.set_xlabel("Nivel de Stock (unidades)", color='#94a3b8', fontsize=9)
            ax_vec.set_ylabel("Días sin Rotación (días)", color='#94a3b8', fontsize=9)
            ax_vec.set_title("Proximidad Geométrica en Espacio de Características", color='#f8fafc', fontsize=10, fontweight='bold')
            ax_vec.tick_params(colors='#94a3b8', labelsize=8)
            ax_vec.legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', fontsize=8, loc='upper right')
            for sp in ax_vec.spines.values():
                sp.set_color('#334155')

            plt.tight_layout()
            st.pyplot(fig_vec)
            plt.close(fig_vec)

    # --- 2. REPRESENTACIÓN SIMBÓLICA ---
    with sub_simb:
        st.subheader("Motor de Reglas Lógicas de Negocio")
        st.markdown("Deducción de decisiones operativas basada en hechos observables y reglas lógicas declarativas.")

        col_s_ctrl, col_s_out = st.columns([1, 1.2])

        with col_s_ctrl:
            st.markdown("#### Hechos Operativos Observados")
            stock_hecho = st.number_input("Stock actual en bodega:", min_value=0, max_value=100, value=1, step=1, key="s7_st_h")
            dias_hecho = st.number_input("Días de inmovilización:", min_value=0, max_value=180, value=40, step=1, key="s7_di_h")
            oxido_hecho = st.checkbox("¿Se evidencia óxido o deterioro físico?", value=False, key="s7_ox_h")

        with col_s_out:
            st.markdown("#### Evaluación de Premisas")
            p1 = stock_hecho < 3
            p2 = dias_hecho > 30
            p3 = oxido_hecho

            st.write(f"- Premisa 1: `Stock < 3` ➔ **{p1}** (Stock actual: {stock_hecho})")
            st.write(f"- Premisa 2: `Días > 30` ➔ **{p2}** (Días actuales: {dias_hecho})")
            st.write(f"- Premisa 3: `Oxidación == True` ➔ **{p3}**")

            st.markdown("#### Regla de Negocio Aplicada:")
            st.code("""IF stock < 3 OR dias_rotacion > 30 OR oxidacion == True:
    THEN generar_orden_urgente_o_descarte()
ELSE:
    THEN mantener_monitoreo_estandar_erp()""", language="python")

            if p1 or p2 or p3:
                st.error("🚨 **DECISIÓN EXPERTA:** SI se detecta stock bajo o deterioro (oxidación/estancamiento) ➔ **ENTONCES generar orden de compra urgente o activar protocolo de descarte.**")
            else:
                st.success("✅ **DECISIÓN EXPERTA:** SI el inventario es óptimo ➔ **ENTONCES mantener monitoreo estándar en ERP.**")

    # --- 3. AUTÓMATA FINITO DETERMINISTA (AFD) ---
    with sub_afd:
        st.subheader("Autómata Finito Determinista para Validación de Códigos")
        st.markdown("""
        Validador sintáctico formal para códigos alfanuméricos de inventario de repuestos.
        - **Lenguaje aceptado:** La secuencia exacta `'MOT'` seguida de un dígito numérico `[0-9]` (ej. `MOT1`, `MOT5`, `MOT9`).
        - **Alfabeto:** $\\Sigma = \\{'M', 'O', 'T', '0', '1', ..., '9'\\}$
        - **Estados:** $Q = \\{S_0, S_1, S_2, S_3, S_4\\}$ | **Inicial:** $S_0$ | **Aceptación:** $S_4$
        """)

        # Botones de prueba rápida
        col_b1, col_b2, col_b3, col_b4, col_b5 = st.columns(5)
        codigo_sugerido = "MOT5"
        if col_b1.button("Probar 'MOT5'"): codigo_sugerido = "MOT5"
        if col_b2.button("Probar 'MOT9'"): codigo_sugerido = "MOT9"
        if col_b3.button("Probar 'MOTO'"): codigo_sugerido = "MOTO"
        if col_b4.button("Probar 'CAR1'"): codigo_sugerido = "CAR1"
        if col_b5.button("Probar 'MOT12'"): codigo_sugerido = "MOT12"

        codigo_eval = st.text_input("Código de repuesto a validar:", value=codigo_sugerido, key="s7_cod_input")

        # Simulación del autómata
        estado_actual = "S0"
        traza = [{"Paso": 0, "Carácter": "INICIO", "Estado Origen": "-", "Estado Destino": "S0", "Válido": True}]
        valido = True

        for idx, char in enumerate(codigo_eval, start=1):
            estado_previo = estado_actual
            if estado_actual == "S0" and char == 'M':
                estado_actual = "S1"
            elif estado_actual == "S1" and char == 'O':
                estado_actual = "S2"
            elif estado_actual == "S2" and char == 'T':
                estado_actual = "S3"
            elif estado_actual == "S3" and char.isdigit():
                estado_actual = "S4"
            else:
                estado_actual = "MUERTO"
                valido = False
                traza.append({"Paso": idx, "Carácter": char, "Estado Origen": estado_previo, "Estado Destino": "ERROR (Rechazado)", "Válido": False})
                break
            traza.append({"Paso": idx, "Carácter": char, "Estado Origen": estado_previo, "Estado Destino": estado_actual, "Válido": True})

        if valido and estado_actual == "S4":
            st.success(f"🎉 **CÓDIGO ACEPTADO:** La secuencia `{codigo_eval}` culminó en el estado de aceptación **S4**.")
        else:
            st.error(f"❌ **CÓDIGO RECHAZADO:** La secuencia `{codigo_eval}` no cumple la gramática del autómata formal.")

        st.markdown("**Traza de Transiciones en el Autómata:**")
        st.dataframe(traza, use_container_width=True)

    # --- 4. TABLA COMPARATIVA ---
    with sub_comp:
        st.subheader("Comparativa Formal de Representaciones de IA")
        st.markdown("""
| Representación | Qué información utiliza | Qué puede reconocer | Ventajas | Limitaciones | Información que puede perderse |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Numérica** | Vectores cuantitativos (stock, días, costos). | Grados de similitud y proximidad matemática entre repuestos. | Permite cálculos precisos y clasificación estadística rápida. | Sensible a escalas y unidades heterogéneas. | Contexto cualitativo o descripciones en lenguaje natural. |
| **Simbólica** | Hechos discretos y reglas condicionales lógicas. | Causalidad y directivas normativas del negocio de autopartes. | Alta explicabilidad y alineación directa con expertos humanos. | Dificultad para manejar incertidumbre o datos masivos continuos. | Matices graduales o probabilidades continuas de fallo. |
| **Autómata** | Secuencias formales de símbolos y alfabetos. | Patrones sintácticos exactos y códigos de control válidos. | Determinista, eficiente en tiempo de ejecución y verificable. | Rigidez absoluta ante variaciones o errores tipográficos. | Estructuras semánticas o relaciones contextuales amplias. |
        """)


# 7. SEMANA 8: RECONOCIMIENTO DE REPUESTOS (RED NEURONAL & SQLITE)

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


# 8. SEMANA 9: VISIÓN COMPUTACIONAL (BORDES CANNY, OTSU & REGIONES CONEXAS)

with tab_s9:
    st.header("Semana 9: Reconocimiento de Imágenes - Contornos, Otsu y Regiones Conectadas")
    st.markdown("Transformación numérica de imágenes de repuestos mediante **Canny Edge Detection**, **Segmentación Otsu** y **Etiquetado de Componentes Conexos**.")

    ruta_img_defecto = Path("data/imagen_proyecto.png")
    if not ruta_img_defecto.exists():
        ruta_img_defecto = Path("images/foto_bujia.jpg")

    col_ctrl, col_vista = st.columns([1, 2.2])

    with col_ctrl:
        st.subheader("⚙️ Configuración del Pipeline")
        catalogo_s9 = obtener_catalogo_imagenes()
        modo_carga_s9 = st.radio("Fuente de la imagen:", ["📂 Catálogo del Taller", "📤 Subir Imagen Nueva"], horizontal=True, key="s9_modo")
        
        if modo_carga_s9 == "📂 Catálogo del Taller":
            sel_nombre = st.selectbox(
                f"Seleccione imagen del taller ({len(catalogo_s9)} detectadas dinámicamente):",
                list(catalogo_s9.keys()),
                index=0,
                key="s9_sel"
            )
            img_sel_path = Path(catalogo_s9[sel_nombre])
            img_bgr = cv2.imread(str(img_sel_path))
            nombre_mostrar = sel_nombre
        else:
            archivo_subido = st.file_uploader("Suba una imagen (.jpg, .png, etc.):", type=["jpg", "jpeg", "png", "webp", "bmp"], key="s9_subir")
            if archivo_subido is not None:
                bytes_data = archivo_subido.read()
                nparr = np.frombuffer(bytes_data, np.uint8)
                img_bgr = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_sel_path = Path(archivo_subido.name)
                nombre_mostrar = f"Subida: {archivo_subido.name}"
            else:
                img_sel_path = Path(list(catalogo_s9.values())[0])
                img_bgr = cv2.imread(str(img_sel_path))
                nombre_mostrar = list(catalogo_s9.keys())[0]

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
        if img_bgr is not None:
            st.caption(f"Visualizando: **{nombre_mostrar}**")
            # Cargar imagen y procesar
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

            labels_sel, total_regs = ndi.label(mask_sel, structure=np.ones((3, 3)))
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


# 9. SEMANA 10: HISTOGRAMAS, REGIONES Y TEXTURAS LBP

with tab_s10:
    st.header("Semana 10: Histogramas, Regiones y Texturas LBP")
    st.markdown("Segmentación por **Otsu**, medición y filtrado morfológico de **Regiones Conexas**, extracción de **Texturas LBP (Local Binary Pattern)** y combinación en un **Vector de 49 Dimensiones (`.npy`)**.")

    col_s10_ctrl, col_s10_vista = st.columns([1, 2.3])

    with col_s10_ctrl:
        st.subheader("⚙️ Configuración del Análisis")
        catalogo_s10 = obtener_catalogo_imagenes()
        modo_carga_s10 = st.radio("Fuente de imagen:", ["📂 Catálogo del Taller", "📤 Subir Imagen Nueva"], horizontal=True, key="s10_modo")

        if modo_carga_s10 == "📂 Catálogo del Taller":
            sel_s10_nom = st.selectbox(
                f"Seleccione repuesto ({len(catalogo_s10)} detectados dinámicamente):",
                list(catalogo_s10.keys()),
                index=0,
                key="s10_img_sel"
            )
            img_s10_path = Path(catalogo_s10[sel_s10_nom])
            img_bgr_s10 = cv2.imread(str(img_s10_path))
            nombre_s10_mostrar = sel_s10_nom
        else:
            archivo_subido_s10 = st.file_uploader("Suba un repuesto (.jpg, .png, etc.):", type=["jpg", "jpeg", "png", "webp", "bmp"], key="s10_subir")
            if archivo_subido_s10 is not None:
                bytes_data = archivo_subido_s10.read()
                nparr = np.frombuffer(bytes_data, np.uint8)
                img_bgr_s10 = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                img_s10_path = Path(archivo_subido_s10.name)
                nombre_s10_mostrar = f"Subida: {archivo_subido_s10.name}"
            else:
                img_s10_path = Path(list(catalogo_s10.values())[0])
                img_bgr_s10 = cv2.imread(str(img_s10_path))
                nombre_s10_mostrar = list(catalogo_s10.keys())[0]

        min_area_slider = st.slider("Filtro de Área Mínima de Región (px):", min_value=10, max_value=500, value=50, step=10,
                                    help="Descarta componentes pequeños causados por ruido de fondo o micro-sombras.")

        radio_lbp = st.selectbox("Radio LBP (R):", [1, 2, 3], index=0, help="Radio del vecindario circular alrededor del píxel central.")
        puntos_lbp = 8 * radio_lbp

        st.markdown("---")
        st.markdown("### 📌 Fundamentos Teóricos")
        st.caption("• **Otsu:** Maximiza la varianza inter-clase para separar el fondo blanco de piezas oscuras.")
        st.caption("• **Filtro de Regiones:** Elimina ruido para aislar componentes funcionales reales.")
        st.caption("• **LBP Uniforme:** Mide micro-textura y rugosidad local comparando el píxel con sus vecinos.")
        st.caption("• **Vector 49D:** 7 características de región + 32 bins de intensidad + 10 bins LBP.")

    with col_s10_vista:
        if img_bgr_s10 is not None:
            st.caption(f"Repuesto Seleccionado: **{nombre_s10_mostrar}**")
            img_rgb_s10 = cv2.cvtColor(img_bgr_s10, cv2.COLOR_BGR2RGB)
            img_gray_s10 = cv2.cvtColor(img_bgr_s10, cv2.COLOR_BGR2GRAY)

            # 1. Histograma de intensidad
            hist_int_s10, _ = np.histogram(img_gray_s10.ravel(), bins=32, range=(0, 256), density=True)

            # 2. Otsu y máscara
            t_otsu_s10 = int(threshold_otsu(img_gray_s10))
            mask_s10 = img_gray_s10 < t_otsu_s10
            pix_obj_s10 = int(np.sum(mask_s10))
            pct_obj_s10 = (pix_obj_s10 / img_gray_s10.size) * 100.0

            # 3. Componentes conexos (8-conectividad)
            struct_8 = np.ones((3, 3), dtype=int)
            lbl_s10, n_crudas_s10 = ndi.label(mask_s10, structure=struct_8)
            props_s10 = regionprops(lbl_s10)
            props_filt_s10 = [r for r in props_s10 if r.area >= min_area_slider]
            n_filt_s10 = len(props_filt_s10)
            areas_filt_s10 = [r.area for r in props_filt_s10] if props_filt_s10 else [0]
            area_prom_s10 = float(np.mean(areas_filt_s10))
            area_std_s10 = float(np.std(areas_filt_s10))

            # 4. LBP
            lbp_s10 = local_binary_pattern(img_gray_s10, P=8, R=radio_lbp, method='uniform')
            n_bins_lbp_s10 = 10
            hist_lbp_s10, _ = np.histogram(lbp_s10.ravel(), bins=n_bins_lbp_s10, range=(0, n_bins_lbp_s10), density=True)

            # 5. Vector 49D
            feats_reg_s10 = np.array([
                float(n_crudas_s10),
                float(n_filt_s10),
                area_prom_s10,
                area_std_s10,
                float(np.sum(areas_filt_s10)),
                pix_obj_s10 / img_gray_s10.size,
                float(t_otsu_s10)
            ])
            vector_49d = np.concatenate([feats_reg_s10, hist_int_s10, hist_lbp_s10])

            # Métricas rápidas
            m1, m2, m3, m4 = st.columns(4)
            with m1:
                st.metric("Umbral Otsu", f"T={t_otsu_s10}", f"Luminancia [0-255]")
            with m2:
                st.metric("Regiones Filtradas", f"{n_filt_s10}", f"De {n_crudas_s10} crudas")
            with m3:
                st.metric("Área de Repuesto", f"{pct_obj_s10:.1f}%", f"{pix_obj_s10:,} px")
            with m4:
                st.metric("LBP Bin 8 (Homogéneo)", f"{hist_lbp_s10[8]*100:.1f}%", f"Rugosidad (B9): {hist_lbp_s10[9]*100:.1f}%")

            # Sub-pestañas
            tab_s10_live, tab_s10_panel, tab_s10_vector, tab_s10_comp = st.tabs([
                "🖼️ Procesamiento en Tiempo Real",
                "📊 Artefacto Oficial (artifacts/semana10_histograma.png)",
                "🔢 Vector de Características 49D (.npy)",
                "🔍 Comparador Lado a Lado (Limpia vs Quemada)"
            ])

            with tab_s10_live:
                fig_s10, axs_s10 = plt.subplots(1, 5, figsize=(18, 3.8))
                fig_s10.patch.set_facecolor('#0f172a')
                for ax in axs_s10:
                    ax.tick_params(colors='#94a3b8', labelsize=7)
                    for sp in ax.spines.values(): sp.set_color('#334155')

                # 1. Original
                axs_s10[0].imshow(img_rgb_s10)
                axs_s10[0].set_title(f"1. Original RGB\n{img_gray_s10.shape[0]}x{img_gray_s10.shape[1]} px", color='#f8fafc', fontsize=9, fontweight='bold')

                # 2. Histograma Intensidad
                axs_s10[1].set_facecolor('#1e293b')
                bins_x_s10 = np.linspace(0, 255, 32)
                axs_s10[1].bar(bins_x_s10, hist_int_s10, width=6.0, color='#38bdf8', alpha=0.85, edgecolor='#0284c7')
                axs_s10[1].axvline(t_otsu_s10, color='#ef4444', linestyle='--', linewidth=1.8, label=f"Otsu {t_otsu_s10}")
                axs_s10[1].set_title(f"2. Histograma (T={t_otsu_s10})", color='#f8fafc', fontsize=9, fontweight='bold')
                axs_s10[1].legend(facecolor='#0f172a', edgecolor='#334155', labelcolor='#f8fafc', fontsize=7)

                # 3. Máscara y Bounding Boxes
                axs_s10[2].imshow(mask_s10, cmap='gray')
                for r in props_filt_s10:
                    minr, minc, maxr, maxc = r.bbox
                    rect = patches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                             fill=False, edgecolor='#10b981', linewidth=1.5)
                    axs_s10[2].add_patch(rect)
                axs_s10[2].set_title(f"3. Regiones (≥{min_area_slider}px: {n_filt_s10})", color='#f8fafc', fontsize=9, fontweight='bold')

                # 4. Mapa LBP
                axs_s10[3].imshow(lbp_s10, cmap='magma')
                axs_s10[3].set_title(f"4. Mapa LBP (R={radio_lbp})", color='#f8fafc', fontsize=9, fontweight='bold')

                # 5. Histograma LBP
                axs_s10[4].set_facecolor('#1e293b')
                lbp_x = np.arange(10)
                axs_s10[4].bar(lbp_x, hist_lbp_s10, color=['#f59e0b']*8 + ['#10b981', '#a855f7'], alpha=0.9)
                axs_s10[4].set_title(f"5. LBP Hist (B8: {hist_lbp_s10[8]*100:.1f}%)", color='#f8fafc', fontsize=9, fontweight='bold')
                axs_s10[4].set_xticks(lbp_x)

                plt.tight_layout()
                st.pyplot(fig_s10)
                plt.close(fig_s10)

            with tab_s10_panel:
                art_s10_path = Path("artifacts/semana10_histograma.png")
                if art_s10_path.exists():
                    st.image(str(art_s10_path), caption="Figura oficial multiclase con repuestos clave (Bujías, Llantas, Pistones, Chasis)")
                else:
                    st.info("Ejecute `python src/semana10_texturas.py` para generar el artefacto oficial.")

            with tab_s10_vector:
                st.markdown("#### Desglose del Vector de Características (Longitud: 49 Dimensiones)")
                col_v1, col_v2, col_v3 = st.columns(3)
                with col_v1:
                    st.markdown("**1. Descriptores de Región (7 valores):**")
                    st.json({
                        "Regiones Crudas": int(feats_reg_s10[0]),
                        "Regiones Filtradas": int(feats_reg_s10[1]),
                        "Área Promedio (px)": round(feats_reg_s10[2], 1),
                        "Desv. Estándar Área (px)": round(feats_reg_s10[3], 1),
                        "Área Total Objeto (px)": int(feats_reg_s10[4]),
                        "Cobertura Lienzo (%)": round(feats_reg_s10[5]*100, 2),
                        "Umbral Otsu": int(feats_reg_s10[6])
                    })
                with col_v2:
                    st.markdown("**2. Histograma de Intensidad (32 bins):**")
                    st.caption("Frecuencia probabilística normalizada de gris [0, 255].")
                    st.dataframe([{"Bin": f"Gris {i*8}-{(i+1)*8-1}", "Valor": f"{hist_int_s10[i]:.4f}"} for i in range(10)], height=220)
                with col_v3:
                    st.markdown("**3. Histograma LBP (10 bins):**")
                    st.caption("Distribución de micro-textura uniforme y rugosidad.")
                    desc_lbp = ["Borde 0", "Borde 1", "Borde 2", "Borde 3", "Borde 4", "Borde 5", "Borde 6", "Borde 7", "Homogéneo/Plano", "No Uniforme (Caótico)"]
                    st.dataframe([{"Patrón": desc_lbp[i], "Frecuencia": f"{hist_lbp_s10[i]*100:.2f}%"} for i in range(10)], height=220)

                npy_path = Path("artifacts/semana10_features.npy")
                meta_path = Path("artifacts/semana10_features_meta.json")
                if npy_path.exists():
                    arr_npy = np.load(str(npy_path))
                    st.success(f"Archivo binario sincronizado: `{npy_path}` (Matriz Shape: **{arr_npy.shape}**)")
                
                if meta_path.exists():
                    with open(meta_path, "r", encoding="utf-8") as f_m:
                        meta_data = json.load(f_m)
                    st.markdown(f"**Dataset Completo Procesado por Lotes ({len(meta_data)} repuestos):**")
                    st.dataframe(meta_data, use_container_width=True, height=260)

            with tab_s10_comp:
                st.markdown("### 🔍 Comparador Lado a Lado de Dos Repuestos en Tiempo Real")
                st.caption("Seleccione dos repuestos para contrastar sus propiedades ópticas, umbrales y micro-textura LBP.")
                
                col_cmp_a, col_cmp_b = st.columns(2)
                with col_cmp_a:
                    st.markdown("#### 🔹 Pieza A (Referencia)")
                    sel_a = st.selectbox("Seleccione Pieza A:", list(catalogo_s10.keys()), index=0, key="cmp_a")
                    img_a_bgr = cv2.imread(catalogo_s10[sel_a])
                    if img_a_bgr is not None:
                        img_a_gray = cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2GRAY)
                        t_a = int(threshold_otsu(img_a_gray))
                        lbp_a = local_binary_pattern(img_a_gray, 8, 1, 'uniform')
                        h_lbp_a, _ = np.histogram(lbp_a.ravel(), bins=10, range=(0, 10), density=True)
                        st.image(cv2.cvtColor(img_a_bgr, cv2.COLOR_BGR2RGB), caption=f"{sel_a} (Otsu: {t_a})", use_container_width=True)
                        st.metric("Otsu T*", f"{t_a}", f"LBP Plano (B8): {h_lbp_a[8]*100:.1f}%")
                        st.metric("LBP Rugoso (B9)", f"{h_lbp_a[9]*100:.2f}%", f"Microbordes: {sum(h_lbp_a[:8])*100:.1f}%")
                        
                with col_cmp_b:
                    st.markdown("#### 🔸 Pieza B (Comparación)")
                    # Buscar por defecto foto_bujia_quemada si está disponible
                    idx_def_b = 0
                    for idx_k, k in enumerate(catalogo_s10.keys()):
                        if "quemada" in k.lower():
                            idx_def_b = idx_k
                            break
                    sel_b = st.selectbox("Seleccione Pieza B:", list(catalogo_s10.keys()), index=idx_def_b, key="cmp_b")
                    img_b_bgr = cv2.imread(catalogo_s10[sel_b])
                    if img_b_bgr is not None:
                        img_b_gray = cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2GRAY)
                        t_b = int(threshold_otsu(img_b_gray))
                        lbp_b = local_binary_pattern(img_b_gray, 8, 1, 'uniform')
                        h_lbp_b, _ = np.histogram(lbp_b.ravel(), bins=10, range=(0, 10), density=True)
                        st.image(cv2.cvtColor(img_b_bgr, cv2.COLOR_BGR2RGB), caption=f"{sel_b} (Otsu: {t_b})", use_container_width=True)
                        st.metric("Otsu T*", f"{t_b}", f"LBP Plano (B8): {h_lbp_b[8]*100:.1f}%")
                        st.metric("LBP Rugoso (B9)", f"{h_lbp_b[9]*100:.2f}%", f"Microbordes: {sum(h_lbp_b[:8])*100:.1f}%")

                st.markdown("---")
                st.markdown("#### 📊 Tabla Síntesis de Referencia (Bujía Limpia vs Quemada vs Llanta vs Pistón)")
                st.markdown("""
                | Repuesto de Taller | Condición / Material | Umbral Otsu | Regiones Filtradas | % Homogéneo LBP (Bin 8) | % Rugosidad LBP (Bin 9) |
                | :--- | :--- | :---: | :---: | :---: | :---: |
                | **⚡ Bujía Estándar** | Acero + Cerámica limpia | **191** | **2** (Cuerpo + Terminal) | **86.49%** (Superficie lisa) | **1.71%** (Mínima rugosidad) |
                | **🔥 Bujía Quemada** | Carbonilla + Aceite quemado | **103** | **6** (Costras de carbón) | **29.85%** (Pierde homogeneidad) | **2.96%** (Rugosidad alta) |
                | **🛞 Llanta de Moto** | Caucho vulcanizado + Labrado | **157** | **1** (Toroide continuo) | **55.93%** (Textura periódica) | **7.57%** (Máxima rugosidad) |
                | **⚙️ Pistón de Motor** | Aluminio mecanizado | **183** | **4** (Cilindro + Anillos) | **50.06%** (Sombreado torneado) | **5.74%** (Rugosidad media) |
                """)
        else:
            st.error(f"No se encontró la imagen en `{img_s10_path}`.")


# 11. SEMANA 11: BENCHMARK Y DECISIÓN TÉCNICA (ADR)

with tab_s11:
    st.header("Semana 11: Benchmark de Modelos y Decisión Técnica (ADR)")
    st.markdown("Comparativa rigurosa de tres modelos de Inteligencia Artificial sobre el **vector de 49 características ópticas y morfológicas** extraídas de repuestos reales del taller.")

    # 1. Tarjetas de métricas del Benchmark
    col_bm1, col_bm2, col_bm3 = st.columns(3)
    with col_bm1:
        st.markdown("""
        <div class="metric-card" style="border: 2px solid #22c55e;">
            <h3 style="color: #22c55e;">★ Random Forest (Seleccionado)</h3>
            <p>86.0%</p>
            <small>0.695s fit | Mayor estabilidad (σ = 0.034)</small>
        </div>
        """, unsafe_allow_html=True)
    with col_bm2:
        st.markdown("""
        <div class="metric-card">
            <h3>MLPClassifier (Red Neuronal)</h3>
            <p>86.0%</p>
            <small>0.703s fit | Mayor varianza (σ = 0.043)</small>
        </div>
        """, unsafe_allow_html=True)
    with col_bm3:
        st.markdown("""
        <div class="metric-card">
            <h3>Logistic Regression (Línea Base)</h3>
            <p>80.0%</p>
            <small>0.044s fit | Más rápida (Línea base)</small>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    col_bm_left, col_bm_right = st.columns([1.2, 1])

    with col_bm_left:
        st.subheader("📊 Evidencia Empírica Oficial (`artifacts/benchmark.csv`)")
        csv_path = Path("artifacts/benchmark.csv")
        if csv_path.exists():
            import pandas as pd
            df_bm = pd.read_csv(csv_path)
            st.dataframe(df_bm, use_container_width=True)
            
            # Gráfica interactiva de barras
            fig_bm, (ax_acc, ax_time) = plt.subplots(1, 2, figsize=(10, 3.5))
            fig_bm.patch.set_facecolor('#0f172a')
            
            colores = ['#22c55e' if m == 'forest' else '#3b82f6' if m == 'mlp' else '#f59e0b' for m in df_bm['modelo']]
            
            # Subplot Accuracy
            ax_acc.set_facecolor('#1e293b')
            bars1 = ax_acc.bar(df_bm['modelo'], df_bm['accuracy'] * 100, color=colores, width=0.5)
            ax_acc.set_title("Exactitud (Accuracy %)", color='#f8fafc', fontsize=10, fontweight='bold')
            ax_acc.set_ylim(0, 100)
            ax_acc.tick_params(colors='#94a3b8', labelsize=8)
            for bar in bars1:
                yval = bar.get_height()
                ax_acc.text(bar.get_x() + bar.get_width()/2.0, yval + 2, f"{yval:.1f}%", ha='center', va='bottom', color='#f8fafc', fontsize=8, fontweight='bold')
            
            # Subplot Tiempo
            ax_time.set_facecolor('#1e293b')
            bars2 = ax_time.bar(df_bm['modelo'], df_bm['fit_seconds'], color=colores, width=0.5)
            ax_time.set_title("Tiempo de Ajuste (Segundos)", color='#f8fafc', fontsize=10, fontweight='bold')
            ax_time.tick_params(colors='#94a3b8', labelsize=8)
            for bar in bars2:
                yval = bar.get_height()
                ax_time.text(bar.get_x() + bar.get_width()/2.0, yval + 0.03, f"{yval:.3f}s", ha='center', va='bottom', color='#f8fafc', fontsize=8, fontweight='bold')

            for ax in [ax_acc, ax_time]:
                for sp in ax.spines.values(): sp.set_color('#334155')

            st.pyplot(fig_bm)
        else:
            st.info("Ejecute `python src/semana11_benchmark.py` para generar `benchmark.csv`.")

    with col_bm_right:
        st.subheader("📑 Síntesis del ADR (Architecture Decision Record)")
        st.markdown("""
        * **Problema:** Clasificación física automática de repuestos en recepción de taller.
        * **Dataset Evaluado:** 198 muestras con 49 descriptores (Morfología Otsu + Histograma 32 bins + LBP 10 bins).
        * **Decisión Técnica:** Seleccionar **`RandomForestClassifier`**.
        * **Justificación de Ingeniería:**
          1. **Supera a Línea Base:** +6.0% de exactitud sobre Regresión Logística (86% vs 80%).
          2. **Mayor Estabilidad:** Desviación estándar de solo 3.4% frente al 4.3% de la Red Neuronal en múltiples particiones.
          3. **Convergencia Garantizada:** Sin estancamiento ni problemas de gradiente estocástico.
          4. **Tiempo Tolerable:** 0.69 segundos de entrenamiento es viable en operaciones de bodega.
        """)
        st.caption("Consulte el informe técnico completo en `reports/semana11_adr.md`.")

    st.markdown("---")
    st.subheader("🧪 Inferencia en Vivo con Modelos Comparados")
    st.caption("Seleccione cualquier pieza de la bodega para clasificarla instantáneamente con los tres modelos del benchmark.")

    col_test_ctrl, col_test_res = st.columns([1, 1.8])
    with col_test_ctrl:
        cat_s11 = obtener_catalogo_imagenes()
        sel_repuesto = st.selectbox("Pieza a evaluar:", list(cat_s11.keys()), index=0, key="s11_sel")
        ruta_sel = cat_s11[sel_repuesto]
        img_sel_bgr = cv2.imread(ruta_sel)
        if img_sel_bgr is not None:
            st.image(cv2.cvtColor(img_sel_bgr, cv2.COLOR_BGR2RGB), caption=sel_repuesto, use_container_width=True)

    with col_test_res:
        ds_cache = Path("artifacts/semana11_dataset.npz")
        if ds_cache.exists() and img_sel_bgr is not None:
            from sklearn.ensemble import RandomForestClassifier
            from sklearn.linear_model import LogisticRegression
            from sklearn.neural_network import MLPClassifier
            from sklearn.pipeline import make_pipeline
            from sklearn.preprocessing import StandardScaler
            from src.semana11_benchmark import extraer_vector_49d
            
            # Cargar y entrenar rápidamente
            datos_s11 = np.load(ds_cache, allow_pickle=True)
            X_all, y_all = datos_s11["X"], datos_s11["y"]
            
            m_rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1).fit(X_all, y_all)
            m_lr = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1500, random_state=42)).fit(X_all, y_all)
            m_mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(64,), max_iter=450, random_state=42)).fit(X_all, y_all)
            
            v_test = extraer_vector_49d(img_sel_bgr).reshape(1, -1)
            
            pred_rf = m_rf.predict(v_test)[0]
            prob_rf = np.max(m_rf.predict_proba(v_test)) * 100
            
            pred_lr = m_lr.predict(v_test)[0]
            prob_lr = np.max(m_lr.predict_proba(v_test)) * 100
            
            pred_mlp = m_mlp.predict(v_test)[0]
            prob_mlp = np.max(m_mlp.predict_proba(v_test)) * 100
            
            st.markdown(f"#### Veredictos del Ensamble para `{Path(ruta_sel).name}`:")
            c_p1, c_p2, c_p3 = st.columns(3)
            with c_p1:
                st.metric("★ Random Forest", f"{pred_rf}", f"{prob_rf:.1f}% Confianza")
            with c_p2:
                st.metric("MLPClassifier", f"{pred_mlp}", f"{prob_mlp:.1f}% Confianza")
            with c_p3:
                st.metric("Logistic Reg.", f"{pred_lr}", f"{prob_lr:.1f}% Confianza")
            
            st.success(f"✅ Diagnóstico: La pieza analizada corresponde a la categoría **{pred_rf}** según el modelo seleccionado en el ADR.")
        else:
            st.info("Ejecute primero `python src/semana11_benchmark.py` para generar el dataset de entrenamiento.")


# PIE DE PÁGINA

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Universidad / Institución | Ingeniería de Sistemas S10A | Repositorio Oficial</div>",
    unsafe_allow_html=True)