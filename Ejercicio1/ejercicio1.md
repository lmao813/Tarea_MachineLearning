# 📝 Aproximación de la función raíz cuadrada con una red neuronal  
---

## 📌 Descripción del ejercicio
A partir de la figura 6.1 propuesta en clase, se pidió tomar una función matemática (por ejemplo, una raíz cúbica, un seno, etc.) y generar un conjunto de datos. Con este dataset, se debía entrenar un modelo de Machine Learning que aproximara una función diferente, en este caso:  
- 📈 **Aproximar la función raíz cuadrada (√x) usando una red neuronal.**

---

## ⚙️ Desarrollo del ejercicio

1. **Generación del dataset**  
   Se generaron 500 valores `x` distribuidos uniformemente entre 0 y 100. La salida esperada `y` fue calculada como `√x`. Este conjunto sirvió como base para el entrenamiento.

2. **Modelo de aprendizaje**  
   Se implementó una red neuronal **Multilayer Perceptron (MLP)** utilizando `scikit-learn`.  
   - Arquitectura: 2 capas ocultas con 10 neuronas cada una.  
   - Activación: ReLU.  
   - Optimizador: Adam.  
   - Iteraciones: 5000.

3. **Prueba y comparación**  
   Se tomaron 10 nuevos valores aleatorios de `x` y se evaluó el modelo entrenado frente a la función `np.sqrt(x)` para verificar su precisión.

## 📊 Resultados

- La red neuronal fue capaz de **aproximar con muy buena precisión** la función raíz cuadrada, presentando un **bajo error cuadrático medio**.
- Se generó una gráfica comparativa entre la curva real de `√x` y la predicción del modelo, mostrando una alta similitud.

<img width="841" height="548" alt="image" src="https://github.com/user-attachments/assets/6eb58c9f-2547-4bf8-9ccc-fd0c6c8188f9" />
<img width="534" height="287" alt="image" src="https://github.com/user-attachments/assets/cdf8dcb0-b144-4969-835a-c95e6b21ef1c" />

## 💡 Conclusión

El uso de redes neuronales resulta adecuado incluso para tareas matemáticas continuas como la raíz cuadrada.  
Este ejercicio permitió reforzar conceptos de entrenamiento, evaluación y visualización en modelos de aprendizaje supervisado.
