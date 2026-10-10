# Informe Técnico - Semana 10: Reconocimiento de Imágenes
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
El histograma de intensidad modela la densidad de probabilidad $P(g)$ de los niveles de gris $g \in [0, 255]$.  
El método de Otsu calcula el umbral óptimo $T^*$ que maximiza la varianza inter-clase $\sigma_b^2(T) = \omega_0(T) \omega_1(T) [\mu_0(T) - \mu_1(T)]^2$, separando automáticamente el plano de fondo de los componentes del taller.

Dado que la cabina de fotografía posee un fondo blanco de alta luminosidad (intensidades entre 200 y 255), los objetos de interés poseen intensidades menores que el umbral ($g < T^*$), por lo que la máscara binaria se define como:
$$M(x, y) = \begin{cases} 1 & \text{si } I(x, y) < T^* \\ 0 & \text{en otro caso} \end{cases}$$

### 2.2 Comparativa de Segmentación por Imagen

| Parámetro de Segmentación | Bujía de Encendido | Llanta de Motocicleta | Pistón de Aluminio |
| :--- | :---: | :---: | :---: |
| **Dimensiones** | 447 × 447 px | 554 × 554 px | 318 × 730 px |
| **Umbral Otsu ($T^*$)** | **`191`** | **`157`** | **`183`** |
| **Píxeles de Objeto Segmentados** | 13,634 px | 128,577 px | 70,245 px |
| **Cobertura sobre el Lienzo** | **6.82%** | **41.89%** | **30.26%** |

- **Interpretación:** En la llanta, el caucho negro produce un histograma fuertemente bimodal con un umbral más bajo ($T=157$), logrando aislar el 41.89% del lienzo. En la bujía ($T=191$), la cerámica blanca supera el umbral y se funde con el fondo, capturando un 6.82% correspondiente a los metales oscuros.

---

## 3. Etiquetado y Medición Morfológica de Regiones

Utilizando conectividad de 8 vecinos (`measure.label` / `ndi.label`) y `regionprops`, cuantificamos la continuidad física de los componentes antes y después de aplicar un **filtro de área mínima ($A \ge 50$ px)** para remover artefactos y micro-sombras:

| Métrica Morfológica | Bujía (Acero/Cerámica) | Llanta (Caucho Vulcanizado) | Pistón (Aluminio) |
| :--- | :---: | :---: | :---: |
| **Regiones Crudas Totales** | 22 | 30 | 152 |
| **Área Promedio Cruda** | 619.73 px | 619.73 px | 462.14 px |
| **Desviación Estándar Cruda** | 2513.01 px | 23059.67 px | 5395.93 px |
| **Regiones Filtradas ($A \ge 50$ px)** | **`2` macro-regiones** | **`1` macro-región** | **`4` macro-regiones** |
| **Área Promedio Filtrada** | **`6,718.00` px** | **`128,466.00` px** | **`17,454.75` px** |
| **Desviación Estándar Filtrada** | `5,344.00` px | `0.00` px | `28,457.95` px |

### Análisis Físico del Filtrado
1. **Bujía:** El filtro de 50 px eliminó 20 micro-regiones de ruido (sombras en los pliegues del aislador), dejando exactamente **2 componentes funcionales reales**: el cuerpo metálico inferior (rosca + tuerca + electrodo) y el terminal roscado superior.
2. **Llanta:** El filtro eliminó 29 pequeñas islas espurias, consolidando **1 única gran región continua de 128,466 px** correspondiente a la silueta toroidal del neumático.
3. **Pistón:** El filtro conservó **4 regiones macro** que separan el cilindro del pistón, los anillos de compresión y el bulón central.

---

## 4. Análisis de Texturas Mediante Patrones Binarios Locales (LBP)

### 4.1 Fundamento del Operador LBP
El operador LBP (`local_binary_pattern`) con vecindario circular ($P=8$ puntos, Radio $R=1$, método `'uniform'`) analiza para cada píxel central $g_c$ la relación de intensidad con sus 8 vecinos $g_p$:
$$\text{LBP}_{P, R} = \sum_{p=0}^{P-1} s(g_p - g_c) \cdot 2^p, \quad s(x) = \begin{cases} 1 & \text{si } x \ge 0 \\ 0 & \text{en otro caso} \end{cases}$$

Bajo la variante *uniforme*, aquellos patrones con como máximo dos transiciones binarias $0 \rightarrow 1$ o $1 \rightarrow 0$ se agrupan en bins dedicados (bins 0 a 7 para bordes y esquinas, bin 8 para zonas completamente planas/uniformes, y bin 9 para texturas caóticas no uniformes).

### 4.2 Comparación de Texturas entre Bujía y Llanta

| Bin LBP | Significado Morfológico | Bujía (Metal/Cerámica) | Llanta (Caucho Rugoso) | Contraste de Textura |
| :---: | :--- | :---: | :---: | :--- |
| **Bin 8** | **Zonas Planas / Homogéneas** | **86.49%** | **55.93%** | **La bujía es +30.5% más plana/homogénea.** El fondo de estudio y la porcelana no tienen rugosidad. |
| **Bin 1** | Micro-Borde Gradual | 1.26% | 4.52% | La llanta tiene **3.6 veces más** micro-bordes por el labrado de tracción. |
| **Bin 4** | Transición y Bordes Medios | 2.58% | 7.62% | La llanta presenta casi el triple de densidad de transiciones estructurales. |
| **Bin 5** | Bordes Opuestos / Relieves | 3.29% | 7.64% | La llanta tiene relieve bidireccional por las estrías de evacuación de agua. |
| **Bin 9** | Patrones No Uniformes (Rugosidad Caótica) | 1.71% | 7.57% | La llanta tiene **4.4 veces más rugosidad de alta frecuencia** que la bujía. |

- **Conclusión de Textura:** El histograma LBP discrimina de forma contundente la micro-textura: el caucho de llanta tiene una dispersión rica en patrones de borde, esquina y rugosidad irregular (44% de patrones no homogéneos), mientras que la bujía concentra el 86.56% de sus píxeles en patrones planos (fondo y porcelana lisa).

---

## 5. Vector de Características Combinado (NumPy)

### 5.1 Estructura del Vector de 49 Dimensiones
Para cada pieza del taller construimos un vector numérico continuo combinando:
1. **Descriptores de Región (7 valores):** `[num_crudo, num_filt, area_prom_filt, area_std_filt, area_tot_filt, cobertura_objeto, umbral_otsu]`
2. **Histograma de Intensidad (32 valores):** Densidad probabilística normalizada del brillo en 32 bins $[0, 255]$.
3. **Histograma LBP (10 valores):** Frecuencia normalizada de los 10 bins de textura uniforme.
$$\vec{v}_{\text{pieza}} \in \mathbb{R}^{49}$$

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
