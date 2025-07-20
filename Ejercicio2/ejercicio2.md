# 📝 Clasificación con SVM (Support Vector Machines)    

## 📚 Descripción del ejercicio  
Se estudió el algoritmo de Support Vector Machines (SVM) y se aplicó a un problema de clasificación no lineal usando el conjunto de datos sintético `make_moons`. Se comparó el rendimiento de un clasificador SVM con kernel **lineal** frente a uno con kernel **RBF (Radial Basis Function)**, para evidenciar su capacidad de separación en espacios no linealmente separables.

## ⚙️ Implementación  
- Se generaron datos no linealmente separables (`make_moons`) con ruido.
- Se entrenaron dos clasificadores SVM:
  - Uno con kernel `linear`
  - Otro con kernel `rbf`
- Se evaluaron sus predicciones sobre el conjunto de prueba y se graficaron las fronteras de decisión de ambos modelos.

## 📊 Resultados  
- El modelo con kernel **RBF** logró una mayor exactitud que el modelo lineal, dado que los datos no son separables linealmente.
- Se evidenció visualmente la diferencia en la capacidad de cada modelo para trazar una frontera efectiva de decisión.
<img width="590" height="490" alt="image" src="https://github.com/user-attachments/assets/6028f9dc-9d5c-4cb8-a3a0-b7c21ad905df" />
  - Exactitud con SVM Lineal: 0.9000
<img width="590" height="490" alt="image" src="https://github.com/user-attachments/assets/90f243be-e8b3-465e-81a8-160abd2a4c00" />
  - Exactitud con SVM RBF: 0.9667

## 🧠 Conceptos clave  
- El **SVM lineal** busca un hiperplano que separe los datos en dos clases con el mayor margen posible.
- El **SVM con kernel RBF** proyecta los datos en un espacio de mayor dimensión donde sí pueden ser separados linealmente.
- La elección del kernel y sus parámetros (`C`, `gamma`) afecta la flexibilidad y generalización del modelo.


## 📌 Observaciones
Este ejercicio permitió visualizar las ventajas de usar kernels no lineales en problemas reales. También se reforzó el entendimiento de cómo funciona el margen, los vectores de soporte y el efecto de la complejidad del modelo sobre el sobreajuste.
