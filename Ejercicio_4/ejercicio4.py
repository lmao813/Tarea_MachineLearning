"""
Algoritmo de Árboles de Decisión Mejorado

Este módulo implementa un árbol de decisión para clasificación con las siguientes mejoras:
1. Documentación detallada de cada función
2. Validación de parámetros
3. Manejo de casos especiales
4. Métricas de evaluación adicionales
5. Visualización mejorada del árbol

Clase DecisionTree:
    Atributos:
        max_depth (int): Profundidad máxima del árbol
        min_samples_split (int): Mínimo de muestras para dividir un nodo
        criterion (str): Criterio de división ('gini' o 'entropy')
        root (Node): Nodo raíz del árbol
    
    Métodos principales:
        fit(X, y): Construye el árbol a partir de datos de entrenamiento
        predict(X): Realiza predicciones para nuevos datos
        print_tree(): Muestra la estructura del árbol
        get_metrics(y_true, y_pred): Calcula métricas de evaluación

Uso básico:
    >>> from decision_tree import DecisionTree
    >>> clf = DecisionTree(max_depth=3)
    >>> clf.fit(X_train, y_train)
    >>> predictions = clf.predict(X_test)
    >>> clf.print_tree()
"""

import numpy as np
from collections import Counter

class Node:
    """
    Clase que representa un nodo en el árbol de decisión.
    
    Atributos:
        feature (int): Índice de la característica usada para dividir
        threshold (float): Valor umbral para la división
        left (Node): Subárbol izquierdo (valores <= threshold)
        right (Node): Subárbol derecho (valores > threshold)
        value (int): Valor de clase (solo en nodos hoja)
    """
    def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
        self.feature = feature
        self.threshold = threshold
        self.left = left
        self.right = right
        self.value = value
    
    def is_leaf(self):
        """Verifica si el nodo es una hoja (no tiene hijos)."""
        return self.value is not None

class DecisionTree:
    def __init__(self, max_depth=100, min_samples_split=2, criterion='gini'):
        """
        Inicializa el árbol de decisión.
        
        Parámetros:
            max_depth (int): Profundidad máxima del árbol
            min_samples_split (int): Mínimo número de muestras para dividir un nodo
            criterion (str): Criterio para medir la calidad de la división ('gini' o 'entropy')
        """
        self.max_depth = max_depth
        self.min_samples_split = min_samples_split
        self.criterion = criterion
        self.root = None
        
        # Validar parámetros
        if criterion not in ['gini', 'entropy']:
            raise ValueError("Criterion must be either 'gini' or 'entropy'")
        if max_depth < 1:
            raise ValueError("max_depth must be greater than 0")
        if min_samples_split < 2:
            raise ValueError("min_samples_split must be at least 2")

    def _gini(self, y):
        """Calcula la impureza Gini para un conjunto de etiquetas."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return 1 - np.sum([p**2 for p in probabilities if p > 0])
    
    def _entropy(self, y):
        """Calcula la entropía para un conjunto de etiquetas."""
        counts = np.bincount(y)
        probabilities = counts / len(y)
        return -np.sum([p * np.log2(p) for p in probabilities if p > 0])
    
    def _information_gain(self, X, y, feature, threshold):
        """Calcula la ganancia de información para una división dada."""
        parent_impurity = self._gini(y) if self.criterion == 'gini' else self._entropy(y)
        
        left_idxs = X[:, feature] <= threshold
        right_idxs = X[:, feature] > threshold
        y_left, y_right = y[left_idxs], y[right_idxs]
        
        if len(y_left) == 0 or len(y_right) == 0:
            return 0
            
        n = len(y)
        n_left, n_right = len(y_left), len(y_right)
        
        child_impurity = (n_left/n * (self._gini(y_left) if self.criterion == 'gini' else self._entropy(y_left)) + \
                         (n_right/n * (self._gini(y_right) if self.criterion == 'gini' else self._entropy(y_right)))
        
        return parent_impurity - child_impurity
    
    def _best_split(self, X, y):
        """Encuentra la mejor división para un nodo."""
        best_gain = -1
        best_feature, best_threshold = None, None
        
        for feature in range(X.shape[1]):
            thresholds = np.unique(X[:, feature])
            for threshold in thresholds:
                gain = self._information_gain(X, y, feature, threshold)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature
                    best_threshold = threshold
        
        return best_feature, best_threshold
    
    def _build_tree(self, X, y, depth=0):
        """Construye recursivamente el árbol de decisión."""
        # Condiciones de parada
        if depth >= self.max_depth or len(y) < self.min_samples_split or len(np.unique(y)) == 1:
            counter = Counter(y)
            most_common = counter.most_common(1)[0][0]
            return Node(value=most_common)
        
        # Encontrar la mejor división
        feature, threshold = self._best_split(X, y)
        if feature is None:  # No se encontró una división útil
            counter = Counter(y)
            most_common = counter.most_common(1)[0][0]
            return Node(value=most_common)
        
        # Dividir los datos
        left_idxs = X[:, feature] <= threshold
        right_idxs = X[:, feature] > threshold
        
        # Construir subárboles
        left = self._build_tree(X[left_idxs], y[left_idxs], depth+1)
        right = self._build_tree(X[right_idxs], y[right_idxs], depth+1)
        
        return Node(feature, threshold, left, right)
    
    def fit(self, X, y):
        """Entrena el árbol de decisión con los datos proporcionados."""
        # Validar inputs
        if len(X) != len(y):
            raise ValueError("X and y must have the same length")
        if len(X) == 0:
            raise ValueError("Empty dataset")
            
        self.root = self._build_tree(X, y)
    
    def _traverse_tree(self, x, node):
        """Recorre el árbol para hacer una predicción para una sola muestra."""
        if node.is_leaf():
            return node.value
            
        if x[node.feature] <= node.threshold:
            return self._traverse_tree(x, node.left)
        else:
            return self._traverse_tree(x, node.right)
    
    def predict(self, X):
        """Predice las etiquetas de clase para las muestras en X."""
        if self.root is None:
            raise ValueError("The tree has not been trained yet. Call fit() first.")
            
        return np.array([self._traverse_tree(x, self.root) for x in X])
    
    def print_tree(self, node=None, depth=0, feature_names=None):
        """Imprime la estructura del árbol de forma legible."""
        if node is None:
            node = self.root
            if node is None:
                print("Tree not trained yet.")
                return
        
        indent = "    " * depth
        
        if node.is_leaf():
            print(f"{indent}└── Class: {node.value}")
            return
        
        feature_name = f"Feature_{node.feature}" if feature_names is None or node.feature >= len(feature_names) else feature_names[node.feature]
        print(f"{indent}{feature_name} <= {node.threshold:.2f}")
        self.print_tree(node.left, depth+1, feature_names)
        
        print(f"{indent}{feature_name} > {node.threshold:.2f}")
        self.print_tree(node.right, depth+1, feature_names)
    
    def get_metrics(self, y_true, y_pred):
        """Calcula métricas de evaluación: precisión, recall, F1-score."""
        from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
        
        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='weighted')
        recall = recall_score(y_true, y_pred, average='weighted')
        f1 = f1_score(y_true, y_pred, average='weighted')
        
        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1
        }
##Cambios para una Aplicación Práctica
#Módulo de Preprocesamiento:
class DataPreprocessor:
    """Clase para preprocesar datos antes de entrenar el árbol."""
    
    def __init__(self):
        self.feature_names = None
        self.class_names = None
    
    def handle_missing_values(self, X, strategy='mean'):
        """Maneja valores faltantes en los datos."""
        if strategy == 'mean':
            return np.nan_to_num(X, nan=np.nanmean(X, axis=0))
        elif strategy == 'median':
            return np.nan_to_num(X, nan=np.nanmedian(X, axis=0))
        else:
            raise ValueError("Strategy must be 'mean' or 'median'")
    
    def encode_categorical(self, X, categorical_features=None):
        """Codifica características categóricas como numéricas."""
        if categorical_features is None:
            return X
        
        from sklearn.preprocessing import OrdinalEncoder
        encoder = OrdinalEncoder()
        X_cat = encoder.fit_transform(X[:, categorical_features])
        X[:, categorical_features] = X_cat
        return X
    
    def normalize_features(self, X):
        """Normaliza las características al rango [0, 1]."""
        return (X - X.min(axis=0)) / (X.max(axis=0) - X.min(axis=0))
#Interfaz de Usuario Simplificada:
class DecisionTreeApp:
    """Interfaz simplificada para usar el árbol de decisión en aplicaciones."""
    
    def __init__(self):
        self.preprocessor = DataPreprocessor()
        self.tree = DecisionTree()
        self.feature_names = None
        self.class_names = None
    
    def train(self, X, y, feature_names=None, class_names=None, preprocess=True):
        """Entrena el modelo con opciones de preprocesamiento."""
        self.feature_names = feature_names
        self.class_names = class_names
        
        if preprocess:
            X = self.preprocessor.handle_missing_values(X)
            X = self.preprocessor.normalize_features(X)
        
        self.tree.fit(X, y)
    
    def predict(self, X, preprocess=True):
        """Realiza predicciones con opciones de preprocesamiento."""
        if preprocess:
            X = self.preprocessor.handle_missing_values(X)
            X = self.preprocessor.normalize_features(X)
        
        return self.tree.predict(X)
    
    def evaluate(self, X_test, y_test):
        """Evalúa el modelo en datos de prueba."""
        y_pred = self.predict(X_test)
        return self.tree.get_metrics(y_test, y_pred)
    
    def visualize_tree(self):
        """Muestra una representación gráfica del árbol."""
        if self.tree.root is None:
            print("El árbol no ha sido entrenado aún.")
            return
        
        print("\n=== Estructura del Árbol de Decisión ===")
        self.tree.print_tree(feature_names=self.feature_names)
        
        # Para una visualización más avanzada, podríamos integrar con graphviz
        try:
            from sklearn.tree import export_text
            print("\nRepresentación textual del árbol:")
            print(export_text(self.tree, feature_names=self.feature_names))
        except ImportError:
            print("Instala scikit-learn para una mejor visualización.")
#Ejemplo de Uso en una Aplicación:
if __name__ == "__main__":
    # Ejemplo con el dataset Iris
    from sklearn.datasets import load_iris
    from sklearn.model_selection import train_test_split
    
    # Cargar datos
    iris = load_iris()
    X, y = iris.data, iris.target
    feature_names = iris.feature_names
    class_names = iris.target_names
    
    # Dividir datos
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Crear y usar la aplicación
    app = DecisionTreeApp()
    app.train(X_train, y_train, feature_names=feature_names, class_names=class_names)
    
    # Evaluar
    metrics = app.evaluate(X_test, y_test)
    print("\nMétricas de evaluación:")
    for name, value in metrics.items():
        print(f"{name}: {value:.4f}")
    
    # Visualizar árbol
    app.visualize_tree()
    
    # Hacer una predicción
    sample = X_test[0]
    print(f"\nPredicción para muestra: {class_names[app.predict([sample])[0]]}")


'''Mejoras Implementadas

Documentación completa: Cada clase y método está bien documentado con descripciones detalladas.
Validación de parámetros: Se agregaron verificaciones para evitar errores comunes.
Manejo de datos: Clase de preprocesamiento para manejar valores faltantes y variables categóricas.
Métricas de evaluación: Se agregaron precision, recall y F1-score además de accuracy.
Visualización mejorada: Representación más clara del árbol con nombres de características.
Interfaz simplificada: Clase "App" que facilita el uso del algoritmo en aplicaciones reales.
Robustez: Manejo de casos especiales como divisiones que no mejoran la pureza.
