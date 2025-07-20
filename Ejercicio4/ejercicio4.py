import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score

# 1. Cargar el dataset
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names
target_names = iris.target_names

# 2. Dividir en entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 3. Entrenar el modelo (árbol de decisión)
model = DecisionTreeClassifier(max_depth=3, random_state=42)  # max_depth limita la profundidad del árbol
model.fit(X_train, y_train)

# 4. Predecir
y_pred = model.predict(X_test)

# 5. Evaluar
print("Reporte de Clasificación:\n")
print(classification_report(y_test, y_pred, target_names=target_names))
print(f"Precisión: {accuracy_score(y_test, y_pred):.2f}")

# 6. Visualizar el árbol
plt.figure(figsize=(12, 6))
plot_tree(model, feature_names=feature_names, class_names=target_names, filled=True)
plt.title("Árbol de Decisión - Clasificación Iris")
plt.show()
