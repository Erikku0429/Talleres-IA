# Informe Técnico - Semana 10: Reconocimiento de Imágenes
**Asignatura:** Inteligencia Artificial  
**Proyecto:** Sistema Inteligente de Inventarios para Taller de Motos (Visión Computacional)  
**Estudiantes:** Erick Santiago Garcia Sanchez, Oscar David Gutierrez  
**Dataset Escaneado:** 34 imágenes procesadas dinámicamente  
**Fecha de Entrega:** Octubre 2026  

---

## 1. Selección de Imágenes del Proyecto y Dataset Dinámico

Para cumplir con el requerimiento de análisis sobre múltiples repuestos y estados físicos (bujías estándar, bujías quemadas/deterioradas, llantas, pistones, chasis, sillines y suspensiones), el sistema implementa un **módulo de descubrimiento dinámico de imágenes**:
* Se escanea automáticamente el directorio `images/` detectando cualquier archivo `.jpg`, `.jpeg`, `.png`, `.bmp` o `.webp`.
* **No requiere modificar código** para añadir nuevas imágenes: basta con colocarlas en `images/` y tanto el pipeline por lotes como la aplicación interactiva Streamlit las detectan, procesan y grafican al instante.
* En total se procesaron **34 imágenes**, combinando estados de desgaste, geometrías y materiales del taller.

### Casos de Estudio Principales para Análisis Comparativo:
1. **Bujía Estándar Limpia (`images/foto_bujia.jpg`):** Cuerpo roscado de acero mecanizado, hexágono y aislador cerámico blanco liso.
2. **Bujía Quemada / Deteriorada (`images/foto_bujia_quemada.jpg`):** Pieza con depósitos de carbonilla, aceite quemado y electrodo erosionado por sobrecalentamiento.
3. **Llanta de Motocicleta (`images/foto_llanta.jpg`):** Banda de rodadura de caucho negro vulcanizado con estrías direccionales de tracción.
4. **Pistón de Motor (`images/foto_piston.jpg`):** Bloque cilíndrico de aluminio maquinado con ranuras para anillos de compresión.

---

## 2. Segmentación Mediante Histogramas y Umbral Otsu

### 2.1 Justificación Técnica
El histograma de intensidad modela la densidad de probabilidad $P(g)$ de los niveles de gris $g \in [0, 255]$. El método de Otsu calcula el umbral óptimo $T^*$ que maximiza la varianza inter-clase:
$$\sigma_b^2(T) = \omega_0(T) \omega_1(T) [\mu_0(T) - \mu_1(T)]^2$$

Dado que el fondo de inspección es blanco/iluminado, el objeto de interés se aísla mediante:
$$M(x, y) = \begin{cases} 1 & \text{si } I(x, y) < T^* \\ 0 & \text{en otro caso} \end{cases}$$

### 2.2 Comparativa de Casos Clave (Bujía Limpia vs. Bujía Quemada vs. Llanta vs. Pistón)

| Parámetro de Segmentación | Bujía Limpia (`foto_bujia.jpg`) | 🔥 Bujía Quemada (`foto_bujia_quemada.jpg`) | 🛞 Llanta (`foto_llanta.jpg`) | ⚙️ Pistón (`foto_piston.jpg`) |
| :--- | :---: | :---: | :---: | :---: |
| **Dimensiones** | 447 × 447 px | 451 × 680 px | 554 × 554 px | 318 × 730 px |
| **Umbral Otsu ($T^*$)** | **`191`** | **`103`** | **`157`** | **`183`** |
| **Píxeles de Objeto** | 13,634 px | 62,468 px | 128,577 px | 70,245 px |
| **Cobertura en Imagen** | **6.82%** | **20.37%** | **41.89%** | **30.26%** |

> **Interpretación Física Crucial:**  
> La **bujía quemada** presenta un umbral Otsu significativamente menor ($T=103$ frente a $T=191$ de la limpia). Esto se debe a que el hollín, carbón y residuos de combustión oscurecen no solo el electrodo sino también la cerámica circundante, modificando drásticamente la bimodalidad del histograma.

---

## 3. Etiquetado y Medición Morfológica de Regiones

Utilizando conectividad de 8 vecinos (`ndi.label` con matriz $3 \times 3$ de unos) y `regionprops`, se cuantifican las regiones antes y después del **filtro de área mínima ($A \ge 50$ px)**:

| Métrica Morfológica | Bujía Limpia | 🔥 Bujía Quemada | 🛞 Llanta | ⚙️ Pistón |
| :--- | :---: | :---: | :---: | :---: |
| **Regiones Crudas** | 22 | 72 | 30 | 152 |
| **Regiones Filtradas ($A \ge 50$ px)** | **`2` componentes** | **`6` componentes** | **`1` componente** | **`4` componentes** |
| **Área Promedio Filtrada** | `6,718.0` px | `10,369.7` px | `128,466.0` px | `17,454.8` px |
| **Desviación Estándar Área** | `5,344.0` px | `21,318.6` px | `0.0` px | `28,458.0` px |

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
| **Bin 8** | **Zonas Homogéneas / Planas** | **86.49%** | **29.85%** | **55.93%** |
| **Bins 0 a 7** | **Micro-Bordes / Relieves** | 11.80% | 67.19% | **36.50%** |
| **Bin 9** | **Rugosidad Caótica / No Uniforme**| 1.71% | **2.96%** | **7.57%** |

> **Hallazgo Clave de IA:**  
> * La bujía limpia tiene **86.5% de homogeneidad (Bin 8)** debido a su porcelana blanca pulida y superficie metálica impecable.
> * La **bujía quemada** se desploma a **29.8% en el Bin 8**, multiplicando sus micro-bordes y patrones no uniformes debido a las costras irregulares de carbón y corrosión térmica.
> * La **llanta** mantiene una respuesta superior de micro-bordes (36.5%) debido al labrado direccional de tracción.

---

## 5. Vector de Características Combinado (49 Dimensiones) y Matriz `.npy`

Para cada imagen se extrae:
$$\vec{v}_{\text{pieza}} = [7 \text{ descriptores de región}] \cup [32 \text{ bins de intensidad}] \cup [10 \text{ bins LBP}] = \mathbf{49\text{ dimensiones}}$$

La matriz completa de forma **`(34, 49)`** fue guardada en:
* [**`artifacts/semana10_features.npy`**](file:///c:/Users/Erick/Documents/GitHub/Talleres-IA/artifacts/semana10_features.npy)
* Metadatos estructurados en [**`artifacts/semana10_features_meta.json`**](file:///c:/Users/Erick/Documents/GitHub/Talleres-IA/artifacts/semana10_features_meta.json)

---

## 6. Inventario Completo del Dataset Procesado (34 Repuestos)

| Archivo | Repuesto / Estado | Dimensiones | Otsu (T) | Cobertura | Regiones (Crud/Filt) | Área Promedio | LBP Plano (B8) | LBP Rugoso (B9) |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| `imagen_proyecto.png` | Bujía del Proyecto (imagen_proyecto.png) | 447x447 | **191** | 6.8% | 22 / **2** | 6,718 px | 86.5% | 1.7% |
| `bug2.jpg` | Bujía #2 (bug2.jpg) | 554x554 | **169** | 14.3% | 63 / **17** | 2,572 px | 74.5% | 2.8% |
| `bug3.jpg` | Bujía #3 (bug3.jpg) | 335x597 | **158** | 16.9% | 153 / **15** | 2,201 px | 62.3% | 5.8% |
| `bug4.jpg` | Bujía #4 (bug4.jpg) | 597x514 | **187** | 9.9% | 8 / **2** | 15,217 px | 83.8% | 1.4% |
| `bug5.jpg` | Bujía #5 (bug5.jpg) | 428x716 | **163** | 20.9% | 17 / **5** | 12,789 px | 70.4% | 2.3% |
| `cha1.jpg` | Chasis / Cadena #1 (cha1.jpg) | 414x738 | **146** | 14.9% | 9 / **6** | 7,606 px | 65.5% | 3.6% |
| `cha2.jpg` | Chasis / Cadena #2 (cha2.jpg) | 452x678 | **159** | 28.3% | 112 / **9** | 9,594 px | 52.6% | 5.3% |
| `cha3.jpg` | Chasis / Cadena #3 (cha3.jpg) | 449x682 | **163** | 11.7% | 46 / **4** | 8,946 px | 80.6% | 3.4% |
| `cha4.jpg` | Chasis / Cadena #4 (cha4.jpg) | 449x682 | **147** | 14.2% | 11 / **2** | 21,660 px | 76.5% | 3.2% |
| `cha5.jpg` | Chasis / Cadena #5 (cha5.jpg) | 479x640 | **60** | 80.6% | 232 / **34** | 7,238 px | 63.2% | 4.3% |
| `foto_bujia.jpg` | Bujía Estándar (foto_bujia.jpg) | 447x447 | **191** | 6.8% | 22 / **2** | 6,718 px | 86.5% | 1.7% |
| `foto_bujia_quemada.jpg` | Bujía Quemada (foto_bujia_quemada.jpg) | 451x680 | **103** | 20.4% | 72 / **6** | 10,370 px | 29.8% | 3.0% |
| `foto_llanta.jpg` | Llanta / Rueda Estándar (foto_llanta.jpg) | 554x554 | **157** | 41.9% | 30 / **1** | 128,466 px | 55.9% | 7.6% |
| `foto_manubrio.jpg` | Manubrio (foto_manubrio.jpg) | 447x447 | **156** | 4.7% | 1 / **1** | 9,410 px | 90.6% | 1.6% |
| `foto_piston.jpg` | Pistón Estándar (foto_piston.jpg) | 318x730 | **183** | 30.3% | 152 / **4** | 17,455 px | 50.1% | 5.7% |
| `pist2.jpg` | Pistón #2 (pist2.jpg) | 346x519 | **200** | 28.4% | 79 / **1** | 50,793 px | 60.2% | 5.2% |
| `pist3.jpg` | Pistón #3 (pist3.jpg) | 424x724 | **174** | 20.3% | 26 / **2** | 31,164 px | 71.7% | 3.0% |
| `pist4.jpg` | Pistón #4 (pist4.jpg) | 678x452 | **145** | 16.0% | 63 / **40** | 1,215 px | 75.2% | 5.0% |
| `pist5.jpg` | Pistón #5 (pist5.jpg) | 504x608 | **170** | 12.6% | 7 / **2** | 19,292 px | 81.7% | 2.7% |
| `rueda2.jpg` | Llanta / Rueda #2 (rueda2.jpg) | 554x554 | **140** | 45.7% | 73 / **1** | 140,251 px | 59.5% | 6.4% |
| `rueda3.jpg` | Llanta / Rueda #3 (rueda3.jpg) | 554x554 | **151** | 28.1% | 62 / **1** | 86,113 px | 69.8% | 5.3% |
| `rueda4.jpg` | Llanta / Rueda #4 (rueda4.jpg) | 554x554 | **171** | 37.6% | 54 / **4** | 28,760 px | 54.3% | 6.0% |
| `rueda5.jpg` | Llanta / Rueda #5 (rueda5.jpg) | 547x365 | **150** | 46.6% | 41 / **1** | 92,939 px | 49.3% | 5.6% |
| `silla1.jpg` | Sillín / Asiento #1 (silla1.jpg) | 554x554 | **157** | 34.4% | 121 / **20** | 5,201 px | 52.3% | 6.1% |
| `silla2.jpg` | Sillín / Asiento #2 (silla2.jpg) | 480x638 | **127** | 40.0% | 17 / **9** | 13,599 px | 25.8% | 4.7% |
| `silla3.jpg` | Sillín / Asiento #3 (silla3.jpg) | 429x713 | **132** | 60.6% | 500 / **3** | 61,385 px | 13.0% | 13.2% |
| `silla4.jpg` | Sillín / Asiento #4 (silla4.jpg) | 739x415 | **128** | 39.0% | 69 / **7** | 17,022 px | 28.6% | 6.3% |
| `silla5.jpg` | Sillín / Asiento #5 (silla5.jpg) | 300x300 | **154** | 32.1% | 1 / **1** | 28,901 px | 68.0% | 4.4% |
| `susp1.jpg` | Suspensión #1 (susp1.jpg) | 500x400 | **173** | 23.4% | 29 / **7** | 6,667 px | 66.7% | 3.6% |
| `susp2.jpg` | Suspensión #2 (susp2.jpg) | 336x447 | **174** | 12.1% | 71 / **13** | 1,367 px | 70.7% | 3.1% |
| `susp3.jpg` | Suspensión #3 (susp3.jpg) | 447x447 | **150** | 16.1% | 51 / **1** | 31,941 px | 77.7% | 3.1% |
| `susp4.jpg` | Suspensión #4 (susp4.jpg) | 450x598 | **150** | 20.2% | 32 / **3** | 18,032 px | 72.6% | 3.2% |
| `susp5.jpg` | Suspensión #5 (susp5.jpg) | 485x631 | **156** | 14.7% | 23 / **8** | 5,628 px | 69.2% | 2.0% |
| `susp6.jpg` | Suspensión #6 (susp6.jpg) | 300x207 | **182** | 23.3% | 31 / **3** | 4,743 px | 53.7% | 5.9% |

---

## 7. Aplicación Futura en el Proyecto
1. **Diagnóstico Automático de Bujías:** Clasificar bujías en "Operativa" vs "Quemada / Carbonizada" evaluando la caída del Bin 8 y el umbral de Otsu.
2. **Inspección de Desgaste de Llantas:** Monitorear la reducción en la densidad de micro-bordes LBP para predecir cuándo el labrado de la llanta está liso y requiere cambio.
