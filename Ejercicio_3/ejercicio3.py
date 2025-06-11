import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

data = {'Masa': [50, 80, 90, 45, 60],
        'Altura': [1.48, 1.82, 1.85, 1.55, 1.60],
        'Genero': ['m', 'h', 'h', 'm', 'm']}
df = pd.DataFrame(data)
punto_nuevo = pd.DataFrame({'Masa': [70], 'Altura': [1.82]})
def calcular_distancias(df, punto_nuevo):
    distancias = []
    for i in range(len(df)):
        distancia = np.sqrt(
            (df.iloc[i]['Masa'] - punto_nuevo['Masa'][0])**2 +
            (df.iloc[i]['Altura'] - punto_nuevo['Altura'][0])**2
        )
        distancias.append((distancia, df.iloc[i]['Genero']))
    return distancias

distancias = calcular_distancias(df, punto_nuevo)
print("Distancias calculadas:", distancias)
def predecir_genero(distancias, k=3):
    distancias_ordenadas = sorted(distancias, key=lambda x: x[0])
    k_vecinos = distancias_ordenadas[:k]
    conteo = {'h': 0, 'm': 0}
    for _, genero in k_vecinos:
        conteo[genero] += 1
    return max(conteo, key=conteo.get)

prediccion_manual = predecir_genero(distancias, k=3)
print("Predicción manual:", prediccion_manual)
from sklearn.neighbors import KNeighborsClassifier

# Entrenar modelo
X = df[['Masa', 'Altura']]
y = df['Genero']
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)

# Predecir
prediccion_sklearn = knn.predict(punto_nuevo)
print("Predicción scikit-learn:", prediccion_sklearn[0])
plt.figure(figsize=(8, 6))
plt.scatter(df.loc[df['Genero'] == 'h', 'Masa'], 
            df.loc[df['Genero'] == 'h', 'Altura'], 
            c='red', label='Hombre')
plt.scatter(df.loc[df['Genero'] == 'm', 'Masa'], 
            df.loc[df['Genero'] == 'm', 'Altura'], 
            c='blue', label='Mujer')
plt.scatter(punto_nuevo['Masa'], punto_nuevo['Altura'], 
            c='black', marker='x', s=100, label='Punto nuevo')
plt.xlabel('Masa (kg)')
plt.ylabel('Altura (m)')
plt.title('Clasificación con KNN (k=3)')
plt.legend()
plt.grid()
plt.show()
for k in [1, 5]:
    prediccion = predecir_genero(distancias, k)
    print(f"Predicción con k={k}: {prediccion}")
