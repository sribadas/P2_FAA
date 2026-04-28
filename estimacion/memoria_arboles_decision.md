# Árboles de Decisión — Memoria (sección)

Objetivo
- Evaluar el rendimiento de clasificadores basados en Árboles de Decisión para la clasificación de niveles de obesidad en el dataset proporcionado.

Metodología
- Preprocesado: se utilizan las mismas columnas numéricas, binarias y categóricas que en el resto de experimentos; las variables categóricas se codifican con one-hot y las binarias como 0/1.
- Validación: validación cruzada 10-fold con normalización Min-Max ajustada únicamente sobre el conjunto de entrenamiento de cada fold.
- Modelo: clasificador Decision Tree (wrapper MLJ/DecisionTree).
- Hiperparámetros: se prueba la profundidad máxima (`max_depth`) en los valores [1, 2, 4, 6, 8, 12, 16] para cubrir desde árboles muy simples hasta relativamente complejos. Se piden al menos 6 valores distintos; hemos probado 7.

Medidas reportadas
- Para cada configuración se calcula el Accuracy medio y su desviación, Train Accuracy medio, F1-score macro (promedio no ponderado entre clases) medio y su desviación, y la tasa de error.
- Además se guarda la matriz de confusión acumulada (suma sobre los 10 folds) para cada configuración.

Formato de resultados (archivos generados)
- `salidas_dt/datos/resultados_dt.csv` : tabla con una fila por configuración y las columnas: Nombre, MaxDepth, Acc_media, Acc_std, TrainAcc_media, TrainAcc_std, F1_media, F1_std, ErrorRate_med, ErrorRate_std.
- `salidas_dt/datos/matrices_confusion_dt.csv` : listado de matrices de confusión acumuladas por configuración (formato similar al usado en SVM/DoME).

Análisis sugerido para la memoria
- Tabla: insertar la tabla `resultados_dt.csv` con orden por F1_media descendente; destacar la mejor configuración (máximo F1 macro).
- Matriz de confusión: mostrar la matriz acumulada de la mejor configuración y comentar las clases con mayor confusión (por ejemplo, sobrepeso vs obesidad leve, etc.).
- Curva de complejidad: graficar F1_macro y Accuracy en función de `max_depth` (línea con bandas de desviación) para mostrar si hay sobreajuste (train acc significativamente mayor que test acc para profundidades altas).
- Interpretación: comentar el efecto de la profundidad en la capacidad del árbol para separar clases. Profundidades bajas tienden a infra-ajustar (baja F1 y alta tasa de error), mientras que profundidades muy altas pueden sobreajustar especialmente si no se controlan otros parámetros (min_samples_leaf, pruning).

Texto propuesto (ejemplos de párrafos para incluir en la memoria)
- "Se realizaron experimentos con clasificadores basados en Árboles de Decisión, evaluando la profundidad máxima del árbol (`max_depth`) como hiperparámetro principal. Se probaron los valores 1, 2, 4, 6, 8, 12 y 16 empleando validación cruzada 10-fold y normalización Min-Max por fold."
- "Los resultados se resumen en la Tabla X. La mejor configuración según F1 macro fue `DT max_depth=NN` (F1 = XX.X%, Accuracy = YY.Y%). La matriz de confusión asociada (Figura Y) muestra que las principales fuentes de error corresponden a ... [aquí interpretar según resultados]."
- "Al analizar la evolución del F1 y del Accuracy respecto a la profundidad, observamos que... [p.ej. mejora hasta profundidad 8 y luego estabilidad/ligero sobreajuste]. Esto sugiere que una profundidad intermedia ofrece un buen balance entre sesgo y varianza para este dataset."
- "Recomendación: emplear la profundidad óptima encontrada (o aplicar poda/pruning y/o restricción en tamaño mínimo de hoja) antes de desplegar el modelo. También considerar ensamblados (Random Forest, Gradient Boosting) si se busca mejorar la robustez frente al ruido y la varianza."

Instrucciones de ejecución
- Ejecutar el script `estimacion/experimentos_arboles.jl` con Julia en el entorno del proyecto; los resultados se escribirán en `salidas_dt/datos/`.
- Comandos sugeridos (desde la carpeta `estimacion`):

```bash
# lanzar experimento en Julia
julia --project=. experimentos_arboles.jl
```

Figuras recomendadas
- Barra/linea con F1_macro y Accuracy vs `max_depth` (incluyendo barras de desviación).
- Matriz de confusión (porcentajes y absolutos) para la mejor configuración.

Notas
- Si desea, puedo añadir automáticamente el código de graficado (script similar a `graficas_resultados_svm.py`) que lea `resultados_dt.csv` y dibuje las figuras mencionadas.
