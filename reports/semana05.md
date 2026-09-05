# Semana 05 - Sistema Híbrido: Sistemas Expertos, TF-IDF y Clasificación
**Proyecto:** Sistema de Monitoreo de Inventario para Tienda de Motos  
**Autores:** Erick Santiago Garcia Sanchez & Oscar David Gutierrez  

## 1. Descripción del Sistema Híbrido
El sistema combina tres componentes fundamentales de la Inteligencia Artificial explicable para la gestión de inventarios:
1. **Motor de Reglas Explicables (Sistema Experto):** Evalúa disparadores de negocio (reorden, promociones, alertas climáticas).
2. **Recuperador de Información (TF-IDF + Similitud Coseno):** Mapea consultas contra la base de conocimiento estructurada (`data/base_conocimiento.txt`).
3. **Clasificador Naive Bayes Supervisado:** Categoriza automáticamente la intención de la solicitud en etiquetas operativas (`RIESGO_REORDEN`, `SOBRESTOCK`, `INCIDENTE_CALIDAD`, `OBSOLESCENCIA`).

## 2. Base de Conocimiento Cargada
Se registraron 8 entradas de dominio específicas para la tienda y taller de motocicletas:

- 1. Los kits de arrastre (cadena, piñón y corona) para motos de 200cc a 400cc requieren revisión de stock semanal y reorden inmediata cuando queden menos de 3 unidades.
- 2. Las pastillas de freno cerámicas y orgánicas para Bajaj Dominar 400 y KTM Duke tienen alta rotación por desgaste preventivo en mantenimientos de 5000 km.
- 3. El aceite sintético 10W-50 y 15W-50 para motor de cuatro tiempos presenta sobrestock si el inventario supera las 50 latas sin rotación en 30 días.
- 4. Los discos de embrague y guayas de acelerador para motocicletas de mensajería urbana sufren falla recurrente por tráfico pesado y uso intensivo.
- 5. Los llantas de compuesto blando y semislick para rodadas en carretera requieren almacenamiento en espacio seco y temperatura controlada para evitar cristalización.
- 6. El sistema eléctrico, baterías de gel y bujías de iridio muestran pico de demanda durante temporadas de lluvia e inundación vial.
- 7. Los repuestos descontinuados o de bajo movimiento (carburadores antiguos, plásticos de modelos >10 años) deben catalogarse para remate o devolución a proveedor.
- 8. El kit de empaquetadura de motor y retenes de suspensión delantera son ítems críticos en mantenimiento mayor de talleres aliados.

## 3. Pruebas de Ejecución y Resultados Explicables

| Consulta | Regla Activada | Evidencia Recuperada | Similitud | Clasificación |
| :--- | :--- | :--- | :---: | :---: |
| El kit de arrastre para Dominar 400 está bajo en stock y se requiere reorden urgente. | `REGLA_1_ACTIVADA` | 1. Los kits de arrastre (cadena, piñón y corona) para motos ... | `0.2695` | **RIESGO_REORDEN** |
| Hay un exceso de aceite sintético 10W-50 acumulado en bodega sin rotación en 30 días. | `REGLA_3_ACTIVADA` | 3. El aceite sintético 10W-50 y 15W-50 para motor de cuatro ... | `0.6292` | **SOBRESTOCK** |
| Se detectaron fallas en el sistema eléctrico y baterías de gel debido a la temporada de lluvia. | `REGLA_4_ACTIVADA` | 6. El sistema eléctrico, baterías de gel y bujías de iridio ... | `0.5976` | **INCIDENTE_CALIDAD** |

### Detalle Explicativo de las Consultas

#### Consulta #1: "El kit de arrastre para Dominar 400 está bajo en stock y se requiere reorden urgente."
- **Regla Activada:** REGLA_1_ACTIVADA: [REORDEN URGENTE] Emitir orden de compra inmediata a proveedor principal de kits de arrastre.
- **Evidencia Recuperada:** 1. Los kits de arrastre (cadena, piñón y corona) para motos de 200cc a 400cc requieren revisión de stock semanal y reorden inmediata cuando queden menos de 3 unidades.
- **Similitud TF-IDF:** `0.2695`
- **Clasificación Predicha:** `RIESGO_REORDEN`
- **Explicación del Resultado:** El sistema identificó palabras clave relativas al estado del ítem y aplicó la regla lógica correspondiente. Al mismo tiempo, la búsqueda TF-IDF aisló la norma de inventario con mayor coincidencia léxico-semántica y Naive Bayes determinó la categoría operativa según los 15 ejemplos de entrenamiento etiquetados.

#### Consulta #2: "Hay un exceso de aceite sintético 10W-50 acumulado en bodega sin rotación en 30 días."
- **Regla Activada:** REGLA_3_ACTIVADA: [PROMOCIÓN / REDISTRIBUCIÓN] Lanzar combo promocional de cambio de aceite con filtro o mover a bodega secundaria.
- **Evidencia Recuperada:** 3. El aceite sintético 10W-50 y 15W-50 para motor de cuatro tiempos presenta sobrestock si el inventario supera las 50 latas sin rotación en 30 días.
- **Similitud TF-IDF:** `0.6292`
- **Clasificación Predicha:** `SOBRESTOCK`
- **Explicación del Resultado:** El sistema identificó palabras clave relativas al estado del ítem y aplicó la regla lógica correspondiente. Al mismo tiempo, la búsqueda TF-IDF aisló la norma de inventario con mayor coincidencia léxico-semántica y Naive Bayes determinó la categoría operativa según los 15 ejemplos de entrenamiento etiquetados.

#### Consulta #3: "Se detectaron fallas en el sistema eléctrico y baterías de gel debido a la temporada de lluvia."
- **Regla Activada:** REGLA_4_ACTIVADA: [ALERTA CLIMÁTICA] Verificar reserva estratégica de baterías de gel y componentes eléctricos ante lluvias.
- **Evidencia Recuperada:** 6. El sistema eléctrico, baterías de gel y bujías de iridio muestran pico de demanda durante temporadas de lluvia e inundación vial.
- **Similitud TF-IDF:** `0.5976`
- **Clasificación Predicha:** `INCIDENTE_CALIDAD`
- **Explicación del Resultado:** El sistema identificó palabras clave relativas al estado del ítem y aplicó la regla lógica correspondiente. Al mismo tiempo, la búsqueda TF-IDF aisló la norma de inventario con mayor coincidencia léxico-semántica y Naive Bayes determinó la categoría operativa según los 15 ejemplos de entrenamiento etiquetados.

## 4. Conclusión
El sistema híbrido cumple exitosamente con la integración requerida para la Semana 5. Permite tomar decisiones automáticas explicables y proporciona soporte de trazabilidad para la gestión de repuestos de motocicletas.