# Reporte Técnico - Semana 8: Representaciones del Reconocimiento
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
