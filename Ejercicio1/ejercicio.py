import numpy as np
import matplotlib.pyplot as plt
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error

# 1. Generar el dataset
X = np.linspace(0, 100, 500).reshape(-1, 1)       # Entradas: valores entre 0 y 100
y = np.sqrt(X).ravel()                            # Salidas: raíz cuadrada de cada valor

# 2. Definir y entrenar la red neuronal
modelo = MLPRegressor(hidden_layer_sizes=(10, 10),  # 2 capas ocultas de 10 neuronas
                      activation='relu',
                      solver='adam',
                      max_iter=5000,
                      random_state=42)
modelo.fit(X, y)

# 3. Probar con 10 valores nuevos
X_test = np.random.uniform(0, 100, 10).reshape(-1, 1)
y_real = np.sqrt(X_test).ravel()
y_pred = modelo.predict(X_test)

# 4. Mostrar resultados
print("x\t\tModelo\t\tRaíz real")
for x, pred, real in zip(X_test.ravel(), y_pred, y_real):
    print(f"{x:.2f}\t→ {pred:.5f}\t| {real:.5f}")

# 5. Visualización
X_full = np.linspace(0, 100, 500).reshape(-1, 1)
y_full_real = np.sqrt(X_full).ravel()
y_full_pred = modelo.predict(X_full)

plt.figure(figsize=(10, 6))
plt.plot(X_full, y_full_real, label="Raíz real (√x)", color="blue")
plt.plot(X_full, y_full_pred, label="Red neuronal", color="orange", linestyle="--")
plt.scatter(X_test, y_real, color="green", label="Valor real (10 puntos)")
plt.scatter(X_test, y_pred, color="red", label="Predicción")
plt.title("Aproximación de √x usando red neuronal")
plt.xlabel("x")
plt.ylabel("√x")
plt.grid(True)
plt.legend()
plt.show()

# 6. Error cuadrático medio
print("\nError cuadrático medio en prueba:", mean_squared_error(y_real, y_pred))
