# Informe Técnico - Semana 9: Reconocimiento de Imágenes
**Asignatura:** Inteligencia Artificial  
**Proyecto:** Sistema Inteligente de Inventarios para Taller de Motos (Visión Computacional)  
**Estudiantes:** Erick Santiago Garcia Sanchez, Oscar David Gutierrez  
**Fecha de Ejecución:** Octubre 2026  

---

## 1. Selección de Imagen y Relación con el Proyecto

### 1.1 Qué representa la imagen
La imagen seleccionada (`data/imagen_proyecto.png`) corresponde a una **bujía de encendido de motocicleta** (referencia estándar para motores de combustión de cuatro tiempos), capturada en el puesto de recepción y control de calidad del taller.

### 1.2 Por qué es útil para el proyecto
En el taller de motos, la bujía es uno de los componentes de más alta rotación y criticidad para el diagnóstico y mantenimiento preventivo. La recepción de repuestos suele ser un proceso manual susceptible a confusiones (por ejemplo, mezclar bujías de rosca larga o corta, o no detectar roscas deformadas y terminales faltantes). Automatizar la inspección visual permite:
1. Validar la llegada de la pieza al módulo de inventario sin intervención humana repetitiva.
2. Extraer características geométricas y estructurales para alimentar los clasificadores de IA desarrollados en la Semana 8.
3. Verificar la integridad física de las secciones funcionales (aislador, rosca y electrodo).

### 1.3 Qué información se espera analizar
- **Fondo vs. Objeto:** Desacoplar la pieza del plano de trabajo blanco para aislar únicamente el componente físico.
- **Límites morfológicos:** Detectar las fronteras entre el electrodo, la rosca mecanizada, la tuerca hexagonal y las nervaduras cerámicas.
- **Distribución de masa visual y conectividad:** Comprobar si las partes metálicas y aislantes conservan continuidad espacial o si se fragmentan en componentes inconexos debido a diferencias de reflectancia.

---

## 2. Extracción de Características Iniciales

| Dimensión Analizada | Valor Obtenido | Justificación e Interpretación Técnica para el Proyecto |
| :--- | :--- | :--- |
| **Dimensiones Espaciales** | 447 × 447 px (3 canales RGB) | Resolución adecuada para capturar detalle de filetes de rosca sin sobrecargar el pipeline computacional en tiempo real. |
| **Relación de Aspecto (Aspect Ratio)** | `1.0000` (Ancho/Alto = 1.0 en lienzo cuadrado, ~0.44 para el bounding box del repuesto) | La esbeltez longitudinal es la característica geométrica primaria que discrimina a las bujías respecto a piezas circulares (llantas) o cilíndricas macizas (pistones). |
| **Intensidad de Gris (Media ± Desv.)** | `244.53 ± 34.07` (Rango: [0, 255]) | La alta media (>240) refleja el predominio del fondo blanco de cabina fotográfica, mientras la desviación estándar captura el contraste oscuro de los componentes de acero y níquel. |
| **Canales de Color (R, G, B)** | R: `244.66` | G: `244.58` | B: `244.50` | Los tres canales son balanceados y casi idénticos, confirmando que la imagen es casi monocromática (metal grisáceo y cerámica blanca), por lo que el color por sí solo no basta y la forma/bordes son obligatorios. |

### Característica más útil para el problema
La **Forma y Densidad de Bordes** resulta ser la característica más determinante. Dado que los metales mecanizados y la cerámica tienen firmas de color neutras (grises y blancos), la red o el clasificador no puede fiarse del tono cromático. La morfología esbelta, combinada con la periodicidad de los bordes de la rosca y las costillas del aislador, constituye la huella dactilar geométrica única de la bujía.

---

## 3. Detección de Contornos con Canny y Efecto del Parámetro Sigma (σ)

El algoritmo Canny opera mediante cuatro etapas fundamentales:
1. Suavizado gaussiano con parámetro σ.
2. Cálculo de gradientes espaciales de intensidad (magnitud y dirección mediante Sobel).
3. Supresión de no-máximos para adelgazar bordes a 1 píxel de espesor.
4. Histéresis de doble umbral para preservar bordes fuertes y conectar bordes débiles válidos.

### 3.1 Comparativa Experimental de Parámetros σ

| Parámetro σ | Píxeles de Borde Activos | Porcentaje de la Imagen | Comportamiento Observado y Análisis Físico |
| :---: | :---: | :---: | :--- |
| **σ = 1.0** | **4,551 px** | **2.28%** | **Máximo nivel de detalle.** Detecta con precisión los filetes microscópicos de la rosca, las micro-arrugas del maquinado metálico, el electrodo de masa y todas las costillas del aislador cerámico. Sin embargo, retiene leve ruido de fondo e irregularidades menores de iluminación. |
| **σ = 2.0** | **3,041 px** | **1.52%** | **Balance óptimo para inspección industrial.** El filtro gaussiano elimina el ruido de alta frecuencia y las texturas superficiales irrelevantes, dejando contornos continuos y nítidos alrededor de la silueta externa y las divisiones de ensamblaje principales. |
| **σ = 3.0** | **1,912 px** | **0.96%** | **Filtro agresivo / Pérdida de características finas.** Al incrementar la dispersión gaussiana, los bordes adyacentes de la rosca se difuminan y fusionan, perdiéndose la definición de los hilos de rosca y el electrodo fino. Solo persiste la silueta macro de la pieza. |

### 3.2 Cómo los bordes permiten encontrar límites entre objeto y fondo
Las transiciones espaciales de intensidad producen picos locales de magnitud de gradiente ||grad(I)|| = sqrt((dI/dx)^2 + (dI/dy)^2). En el perímetro del repuesto, el paso del metal oscuro al fondo blanco genera una magnitud de gradiente sobresaliente que supera con creces el umbral alto de Canny, aislando la frontera física exacta del componente sin depender de variaciones globales de brillo.

---

## 4. Segmentación Mediante Umbral Automático de Otsu

### 4.1 Parámetros y Resultados Numéricos
- **Umbral Otsu Normalizado [0, 1]:** `0.7480`
- **Umbral Otsu Escala 8-bit [0, 255]:** **`191`**
- **Píxeles asignados a Objeto (Primer Plano):** `13,669 px` (6.84% del área total)
- **Píxeles asignados a Fondo:** `186,140 px` (93.16% del área total)

### 4.2 Explicación del Resultado
El método de Otsu busca de forma automática el umbral óptimo T* que maximiza la varianza inter-clase, lo cual equivale a minimizar la varianza intra-clase de las dos poblaciones de píxeles:
$$\sigma_b^2(T) = \omega_0(T) \cdot \omega_1(T) \cdot [\mu_0(T) - \mu_1(T)]^2$$

Al aplicar este umbral sobre nuestra imagen:
1. **Regiones Metálicas:** La tuerca hexagonal, la rosca inferior y el electrodo presentan valores de intensidad significativamente por debajo de 191 (entre 20 y 160), por lo que quedan clasificados de forma limpia y contundente en la máscara binaria.
2. **Aislador Cerámico:** Dado que la cerámica de alúmina es blanca y la cabina de luz es blanca, la intensidad de la porcelana supera en su mayor parte el umbral de 191. Como consecuencia, el cuerpo de porcelana se confunde con el fondo, separando visualmente la bujía en dos componentes aislados (la parte metálica inferior y el terminal metálico superior). Esto pone en evidencia una limitación intrínseca de los métodos de umbralización basados en intensidad global frente a objetos bicolores o multimaterial.

---

## 5. Análisis de Regiones Conectadas

### 5.1 Conteo y Cuantificación
- **Número Total de Regiones Conectadas Detectadas:** **`23` regiones conexas.**

### 5.2 Detalle de Regiones Principales

| Región | ID Etiqueta | Área (px) | Bounding Box `(y_min, x_min, y_max, x_max)` | Centroide `(y, x)` | Interpretación Física en la Pieza |
| :---: | :---: | :---: | :---: | :---: | :--- |
| Región 1 | 19 | 12,091 | `(204, 172, 412, 270)` | `(309.0, 221.0)` | Cuerpo metálico inferior (tuerca hexagonal, rosca y electrodo de masa) |
| Región 2 | 1 | 1,377 | `(44, 209, 100, 244)` | `(70.2, 226.0)` | Terminal metálico superior roscado (conexión a capuchón de alta tensión) |
| Región 3 | 2 | 39 | `(101, 204, 114, 212)` | `(107.1, 206.7)` | Sombra / resalte de costilla del aislador cerámico (Área: 39 px) |
| Región 4 | 5 | 32 | `(115, 201, 126, 211)` | `(120.1, 204.9)` | Sombra / resalte de costilla del aislador cerámico (Área: 32 px) |
| Región 5 | 8 | 31 | `(128, 202, 137, 208)` | `(131.8, 203.9)` | Sombra / resalte de costilla del aislador cerámico (Área: 31 px) |


### 5.3 ¿Corresponden o no a objetos reales?
- **Región 1 (12,091 px):** **SÍ corresponde a un objeto mecánico real.** Representa el conjunto metálico inferior unificado: la tuerca hexagonal para llave de bujías, la zona roscada y el electrodo inferior de masa.
- **Región 2 (1,377 px):** **SÍ corresponde a un objeto mecánico real.** Representa el terminal roscado superior donde encaja el capuchón del cable de alta tensión de la bobina.
- **Regiones 3 a 23 (entre 8 y 39 px):** **NO corresponden a objetos independientes.** Son artefactos lumínicos, micro-sombras proyectadas en las curvaturas de las costillas aislantes del cuerpo cerámico y pequeñas marcas de pintura/impresión de la marca NGK.
- **Conclusión de Regiones:** Físicamente en el mundo real, la bujía es **un único objeto ensamblado**. Sin embargo, la segmentación numérica por intensidad produce 2 macro-regiones funcionales y 21 micro-regiones de ruido, debido a que el cuerpo aislante central posee la misma reflectancia que el fondo blanco.

---

## 6. Cambios Realizados al Modificar Sigma (σ)

Al modificar de forma interactiva y experimental el parámetro σ en el script y en el dashboard:
1. **Con σ < 1.0:** Se obtienen más de 6,000 píxeles de borde. Aparece ruido salt-and-pepper y bordes espurios causados por la rugosidad microscópica del metal.
2. **Con σ = 1.5 - 2.0:** Se alcanza el compromiso óptimo: 3,041 píxeles de borde. La silueta es continua, se conservan los surcos de la rosca y se delimita perfectamente el terminal superior.
3. **Con σ ≥ 3.0:** Los píxeles de borde caen a 1,912 px. Los bordes finos de la rosca desaparecen por sobre-suavizado gaussiano, y los límites del electrodo se distorsionan.

---

## 7. Limitaciones Encontradas

1. **Ambigüedad de Intensidad Cerámica-Fondo:** El método de Otsu asume una distribución bimodal. Al tener una bujía con aislador blanco sobre fondo blanco, el cuerpo cerámico desaparece en la máscara binaria, dividiendo la pieza en dos fragmentos desconectados.
2. **Sensibilidad a Sombras:** Los pliegues del aislador crean sombras de 8 a 39 píxeles que el algoritmo de componentes conexos clasifica erróneamente como regiones independientes.
3. **Pérdida de Información en Binarización Simple:** Otsu colapsa toda la gradación de textura a 0 o 1, perdiendo el relieve interno de la rosca.

---

## 8. Aplicación Futura Dentro del Proyecto

Esta práctica de procesamiento primario de imágenes sienta las bases técnicas para las siguientes evoluciones del sistema de taller:
1. **Corrección de Fondos (Chroma Key o Fondo Contrastante):** Para la estación de captura del taller, se implementará un fondo de alto contraste (ej. tapete verde antirreflejo o azul mate) para que tanto el metal como la cerámica queden contenidos en una única máscara cerrada.
2. **Fusión de Canny + Otsu (Segmentación basada en Gradientes y Watershed):** Utilizar los contornos de Canny para cerrar los límites del cuerpo cerámico y aplicar transformada de distancia (*watershed*) para fusionar las regiones 1 y 2 en una sola entidad física unificada.
3. **Alimentación al Clasificador de la Semana 8:** El vector de características de 4 dimensiones de la semana anterior (RGB + aspect ratio) puede enriquecerse ahora con:
   - Densidad de bordes de Canny (σ=2.0).
   - Relación de áreas entre componentes conexos mayores.
   - Perímetro y compacidad morfológica.
   Esto resolverá el problema de piezas con colores similares detectado en la Semana 8.
