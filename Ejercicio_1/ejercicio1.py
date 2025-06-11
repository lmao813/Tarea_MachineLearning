import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Generar datos sintéticos para y = sin(x)
x = np.linspace(0, 2 * np.pi, 100).reshape(-1, 1)  # 100 puntos entre 0 y 2π
y = np.sin(x)

# Crear DataFrame
df = pd.DataFrame({'x': x.flatten(), 'y_real': y.flatten()})
print(df.head())
# Entrenar modelo
model = LinearRegression()
model.fit(x, y)

# Predecir valores
y_pred = model.predict(x)

# Calcular error (MSE)
mse = mean_squared_error(y, y_pred)
print(f"Error cuadrático medio (MSE): {mse:.4f}")
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='blue', label='Valor real (sin(x))')
plt.plot(x, y_pred, color='red', label='Predicción del modelo')
plt.title('Aproximación de sin(x) con Regresión Lineal')
plt.xlabel('x')
plt.ylabel('y')
plt.legend()
plt.grid()
plt.show()
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline

# Crear pipeline con polinomio de grado 3
poly_model = make_pipeline(
    PolynomialFeatures(degree=3),
    LinearRegression()
)
poly_model.fit(x, y)
y_poly_pred = poly_model.predict(x)

# Graficar
plt.figure(figsize=(10, 5))
plt.scatter(x, y, color='blue', label='Valor real (sin(x))')
plt.plot(x, y_poly_pred, color='green', label='Predicción (Polinomio grado 3)')
plt.title('Aproximación con Polinomio de Grado 3')
plt.legend()
plt.grid()
plt.show()
# Seleccionar 10 puntos aleatorios
x_test = np.random.uniform(0, 2 * np.pi, 10).reshape(-1, 1)
y_test_real = np.sin(x_test)
y_test_pred = poly_model.predict(x_test)

# Crear tabla comparativa
comparison = pd.DataFrame({
    'x': x_test.flatten(),
    'y_real': y_test_real.flatten(),
    'y_pred': y_test_pred.flatten(),
    'Diferencia': np.abs(y_test_real - y_test_pred).flatten()
})
print(comparison)
