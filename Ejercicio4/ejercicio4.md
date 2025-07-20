# 📝 Árboles de Decisión para Clasificación de Flores  

## 🧪 Aplicación práctica  
**Dataset usado:** Iris (incluido en `scikit-learn`)  
**Objetivo:** Clasificar flores entre tres especies: *Setosa*, *Versicolor* y *Virginica*, basándose en cuatro características morfológicas.  
**Entrenamiento:** Se dividió el conjunto de datos en entrenamiento y prueba (80/20).  
**Modificación aplicada:** Se limitó la profundidad del árbol (`max_depth=3`) para evitar sobreajuste y mejorar la interpretabilidad.

## 📈 Resultados  
- Precisión promedio superior al 90% en el conjunto de prueba.  
- Visualización clara del árbol que permite entender cada decisión del modelo paso a paso.  
- Uso de métricas como `classification_report` y `accuracy_score` para validar el desempeño del modelo.

  <img width="651" height="343" alt="image" src="https://github.com/user-attachments/assets/c951015f-4dfd-4493-a906-272b3a73954e" />
  <img width="950" height="504" alt="image" src="https://github.com/user-attachments/assets/8a48870e-f694-41f0-a4ff-1b98adfbcc9c" />


## ✅ Conclusión  
El ejercicio permitió comprender la lógica de los árboles de decisión, cómo configuran divisiones jerárquicas del espacio de entrada y cómo se pueden controlar sus características para mejorar el aprendizaje y evitar sobreajuste.

