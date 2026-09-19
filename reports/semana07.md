# Semana 07 - Representaciones del Reconocimiento
## Proyecto: Monitoreo de Inventario Visual y Gestión para Tienda de Motos

### 1. Introducción
En esta práctica se implementaron y evaluaron tres representaciones del reconocimiento aplicadas a la gestión de inventario de repuestos para motocicletas:
1. **Representación Numérica:** Análisis vectorial de características de stock y distancia euclidiana.
2. **Representación Simbólica:** Motor de reglas lógicas para toma de decisiones en taller.
3. **Autómata Finito:** Validación formal de códigos y patrones de partes en el sistema.

### 2. Implementación de las Representaciones

#### A. Representación Numérica
Se define un vector de características compuesto por `[nivel_stock, dias_sin_rotacion]`. Utilizando la distancia euclidiana, el sistema compara piezas para identificar similitudes en estado crítico.
- **Pieza de Referencia (Crítica):** `[2.0, 35.0]`
- **Distancia a Pieza 1 (`[1.0, 40.0]`):** `5.10` (Alta similitud, requiere atención).
- **Distancia a Pieza 2 (`[45.0, 2.0]`):** `54.20` (Baja similitud, inventario saludable).

#### B. Representación Simbólica
Se establecieron hechos operativos y reglas de negocio adaptadas al almacén de repuestos:
- **Hechos:** Stock actual, días de estancamiento y estado de oxidación de la pieza.
- **Regla Aplicada:** 
  > *SI* stock < 3 *O* días_rotacion > 30 *O* oxidación == True $ightarrow$ *ENTONCES* generar orden de compra urgente o activar protocolo de descarte.
- **Resultado de Evaluación de Prueba:** SI se detecta stock bajo o deterioro (oxidación/estancamiento) -> ENTONCES generar orden de compra urgente o activar protocolo de descarte.

### C. Reconocimiento mediante Autómata Finito
Se diseñó un Autómata Finito Determinista (AFD) para validar los códigos alfanuméricos de control de inventario de las autopartes (ej. formato `MOT` seguido de un dígito de validación).
- **Alfabeto ($\Sigma$):** `{'M', 'O', 'T', [0-9]}`
- **Estados ($Q$):** `{'S0', 'S1', 'S2', 'S3', 'S4'}`
- **Estado Inicial:** `S0`
- **Estados de Aceptación:** `{'S4'}`
- **Pruebas:** 
  - Secuencia `"MOT5"` $ightarrow$ **Aceptada** (`True`).
  - Secuencia `"MOTO"` $ightarrow$ **Rechazada** (`False`).

### 3. Tabla Comparativa de Representaciones

| Representación | Qué información utiliza | Qué puede reconocer | Ventajas | Limitations | Información que puede perderse |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Numérica** | Vectores de valores cuantitativos (stock, días, costos). | Grados de similitud y proximidad matemática entre repuestos. | Permite cálculos precisos y clasificación estadística rápida. | Sensible a escalas y unidades de medida heterogéneas. | Contexto cualitativo o descripciones en lenguaje natural. |
| **Simbólica** | Hechos discretos y reglas condicionales lógicas. | Causalidad y directivas normativas del negocio de autopartes. | Alta explicabilidad y alineación directa con expertos humanos. | Dificultad para manejar incertidumbre o datos continuos masivos. | Matices graduales o probabilidades estadísticas de fallo. |
| **Autómata** | Secuencias formales de símbolos y alfabetos definidos. | Patrones sintácticos exactos y códigos de control válidos. | Determinista, eficiente en tiempo de ejecución y fácil de verificar. | Rigidez absoluta ante variaciones o errores tipográficos. | Estructuras semánticas o relaciones contextuales amplias. |

### 4. Conclusiones
La combinación de las tres representaciones demuestra cómo un sistema inteligente puede abordar el control de inventarios desde diferentes perspectivas: matemática, lógica experta y validación formal de patrones.
