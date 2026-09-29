import os
import sqlite3
import datetime
import xml.etree.ElementTree as ET
import numpy as np
from PIL import Image
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# --- Configuración de rutas ---
os.makedirs("artifacts", exist_ok=True)
os.makedirs("images", exist_ok=True)
os.makedirs("reports", exist_ok=True)
DB_PATH = "artifacts/inventario_evidencia.db"
ONTOLOGY_PATH = "artifacts/ontologia_motos.graphml"
REPORTE_PATH = "reports/semana08.md"

# --- 1. Generar imágenes falsas si no hay reales ---
def asegurar_imagenes_prueba():
    rutas = {
        "images/foto_bujia.jpg": (200, 200, 200),
        "images/foto_piston.jpg": (150, 150, 150),
        "images/foto_llanta.jpg": (30, 30, 30),
        "images/foto_manubrio.jpg": (100, 50, 50) # Pieza trampa
    }
    for ruta, color in rutas.items():
        if not os.path.exists(ruta):
            size = (50, 150) if "bujia" in ruta else (100, 100)
            img = Image.new('RGB', size, color=color)
            img.save(ruta)

# --- 2. Base de Datos ---
def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS evidencia_reconocimiento (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            imagen_analizada TEXT NOT NULL,
            caracteristicas TEXT NOT NULL,
            prediccion_ia TEXT NOT NULL,
            confianza_porcentaje REAL NOT NULL,
            fecha_registro TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    conn.commit()
    conn.close()

# --- 3. Extracción de Características ---
def extraer_caracteristicas(ruta_imagen):
    img = Image.open(ruta_imagen).convert('RGB')
    arr = np.array(img)
    R_promedio = np.mean(arr[:, :, 0])
    G_promedio = np.mean(arr[:, :, 1])
    B_promedio = np.mean(arr[:, :, 2])
    ancho, alto = img.size
    aspect_ratio = ancho / alto if alto > 0 else 1.0
    return [R_promedio, G_promedio, B_promedio, aspect_ratio]

# --- 4. Entrenamiento de la Red Neuronal ---
def entrenar_modelo():
    print("--- 1. Entrenando Modelo de Visión Computacional ---")
    np.random.seed(42)
    
    X_bujias = np.random.normal(loc=[200, 200, 200, 0.3], scale=[10, 10, 10, 0.05], size=(150, 4))
    y_bujias = np.zeros(150)
    X_pistones = np.random.normal(loc=[150, 150, 150, 1.0], scale=[15, 15, 15, 0.1], size=(150, 4))
    y_pistones = np.ones(150)
    X_llantas = np.random.normal(loc=[30, 30, 30, 1.0], scale=[5, 5, 5, 0.1], size=(150, 4))
    y_llantas = np.array([2] * 150)

    X = np.vstack((X_bujias, X_pistones, X_llantas))
    y = np.hstack((y_bujias, y_pistones, y_llantas))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    mlp = MLPClassifier(hidden_layer_sizes=(16, 8), max_iter=1000, random_state=42)
    mlp.fit(X_train, y_train)

    y_pred = mlp.predict(X_test)
    print(f"Dataset: {len(X_train)} imgs entrenamiento, {len(X_test)} imgs validación.")
    print(f"Exactitud (Accuracy): {accuracy_score(y_test, y_pred) * 100:.2f}%\n")
    return mlp

# --- 5. Ejecución, Umbral y Registro ---
def procesar_imagenes(modelo):
    print("--- 2. Procesando Imágenes Hardcodeadas ---")
    mapa_clases = {0: "Bujía", 1: "Pistón", 2: "Llanta"}
    UMBRAL_CONFIANZA = 0.75

    imagenes_a_probar = [
        "images/foto_bujia.jpg", 
        "images/foto_piston.jpg", 
        "images/foto_manubrio.jpg"
    ]

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    for ruta in imagenes_a_probar:
        if not os.path.exists(ruta): continue
        features = extraer_caracteristicas(ruta)
        x_in = np.array(features).reshape(1, -1)
        probabilidades = modelo.predict_proba(x_in)[0]
        confianza_maxima = np.max(probabilidades)
        clase_predicha = np.argmax(probabilidades)
        
        resultado_final = mapa_clases[clase_predicha] if confianza_maxima >= UMBRAL_CONFIANZA else "Pieza_No_Reconocida"

        feat_str = f"R:{features[0]:.0f}, G:{features[1]:.0f}, B:{features[2]:.0f}, AR:{features[3]:.2f}"
        cursor.execute('''
            INSERT INTO evidencia_reconocimiento (imagen_analizada, caracteristicas, prediccion_ia, confianza_porcentaje, fecha_registro)
            VALUES (?, ?, ?, ?, ?)
        ''', (os.path.basename(ruta), feat_str, resultado_final, round(confianza_maxima * 100, 2), datetime.datetime.now()))
        
        print(f"Imagen: {os.path.basename(ruta)} -> IA: {confianza_maxima * 100:.1f}% -> {resultado_final}")

    conn.commit()
    conn.close()
    print("Evidencia guardada en SQLite.\n")

# --- 6. Verificación de Ontología ---
def verificar_ontologia():
    print("--- 3. Verificando Ontología GraphML ---")
    tree = ET.parse(ONTOLOGY_PATH)
    root = tree.getroot()
    ns = {'g': 'http://graphml.graphdrawing.org/xmlns'}
    edges = root.findall('.//g:edge', ns)
    for e in edges:
        data = e.find('g:data', ns)
        rel = data.text if data is not None else "relacionado_con"
        print(f" - {e.attrib['source']} --[{rel}]--> {e.attrib['target']}")
    print("")

# --- 7. Generación Automática del Reporte ---
def generar_reporte_tecnico():
    print("--- 4. Generando Reporte Técnico Automático ---")
    contenido_markdown = """# Reporte Técnico - Semana 8: Representaciones del Reconocimiento
**Proyecto:** Sistema Inteligente de Inventarios para Taller de Motos (Visión Computacional)

## 1. Qué reconoce el sistema
El modelo reconoce visualmente el **Tipo de Repuesto** a partir de una imagen individual ingresada al sistema. 
Actualmente clasifica tres componentes base: `Bujía`, `Pistón` y `Llanta`. Además, incorpora un mecanismo de exclusión que etiqueta como `Pieza_No_Reconocida` cualquier repuesto que no pertenezca al dominio entrenado (ej. un manubrio).

## 2. Cómo funciona el modelo
- **Arquitectura:** Red Neuronal Artificial (`MLPClassifier`).
- **Extracción de Patrones:** El sistema no lee la imagen cruda, sino que extrae un vector numérico de 4 dimensiones: 
  1. Promedio de canal Rojo (R)
  2. Promedio de canal Verde (G)
  3. Promedio de canal Azul (B)
  4. Proporción de aspecto (Ancho/Alto).
- **Validación de Umbral:** Utiliza la función `predict_proba`. Si la certeza de la red neuronal es inferior al 75%, rechaza la clasificación por seguridad.

## 3. Información registrada en la base de datos
Se utiliza una base de datos SQLite (`inventario_evidencia.db`) que audita cada imagen procesada en la tabla `evidencia_reconocimiento`.
Se almacena:
- `imagen_analizada`: Nombre del archivo validado.
- `caracteristicas`: Vector numérico extraído de la foto.
- `prediccion_ia`: Veredicto final tras aplicar el umbral de confianza.
- `confianza_porcentaje`: Nivel de certidumbre de la red.
- `fecha_registro`: Marca temporal de la auditoría.

## 4. Conceptos que representa la ontología
El archivo `ontologia_motos.graphml` define el ecosistema de reconocimiento mediante 5 conceptos y 5 relaciones estructurales:
- `ImagenRepuesto` **procesada_por** `RedNeuronal`
- `RedNeuronal` **predice_categoria** `ClasePieza`
- `RedNeuronal` **valida_certidumbre_de** `EstadoReconocimiento`
- `EstadoReconocimiento` **registra_evidencia_en** `BaseDeDatos`
- `BaseDeDatos` **audita_historial_de** `ImagenRepuesto`

## 5. Limitaciones encontradas
El sistema depende de una correcta iluminación. Piezas estructuralmente distintas pero con el mismo tono de pintura y proporción podrían generar un "falso positivo" si superan el umbral del 75%. Para escalar a un entorno de producción real, este clasificador basado en color/forma debe ser sustituido por una arquitectura convolucional (CNN) orientada a la detección de bordes complejos.
"""
    with open(REPORTE_PATH, "w", encoding="utf-8") as f:
        f.write(contenido_markdown)
    print(f"✅ Reporte guardado exitosamente en: {REPORTE_PATH}\n")

if __name__ == "__main__":
    asegurar_imagenes_prueba()
    init_db()
    modelo = entrenar_modelo()
    procesar_imagenes(modelo)
    verificar_ontologia()
    generar_reporte_tecnico()