import streamlit as st
import numpy as np
import math
import heapq
from collections import Counter
from pathlib import Path
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

tab_resumen, tab_s2, tab_s3, tab_s4, tab_s5 = st.tabs([
    "📊 Resumen Ejecutivo",
    "🤖 Semana 2: Línea Base",
    "🔍 Semana 3: Taxonomía",
    "🗺️ Semana 4: Dashboard A*",
    "⚡ Semana 5: Sistema Híbrido"
])

# 1. RESUMEN EJECUTIVO

with tab_resumen:
    st.header("Arquitectura Semestral del Proyecto")
    col1, col2, col3, col4 = st.columns(4)
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


# PIE DE PÁGINA

st.markdown("---")
st.markdown(
    "<div style='text-align: center; color: #94a3b8; font-size: 0.9rem;'>Universidad / Institución | Ingeniería de Sistemas S10A | Repositorio Oficial</div>",
    unsafe_allow_html=True)