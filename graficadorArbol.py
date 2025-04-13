# graficador_arbol.py
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt
import pandas as pd

def dibujar_arbol(dataset, nivel=None, nombre_archivo='arbol_decision'):
    # Codificar las variables categóricas
    label_encoders = {}
    for column in dataset.columns:
        le = LabelEncoder()
        dataset[column] = le.fit_transform(dataset[column])
        label_encoders[column] = le

    # Separar características (X) y etiquetas (y)
    X = dataset.iloc[:, :-1]  # Todas las columnas excepto la última
    y = dataset.iloc[:, -1]   # Última columna (Resultado)

    print (f"\nDataset después de sklearn:\n{dataset}")  # Mostrar los primeros registros del dataset

    # Crear y entrenar el árbol de decisiones
    clf = DecisionTreeClassifier(criterion='entropy', max_depth=nivel, random_state=42)
    clf.fit(X, y)

    # Visualizar el árbol de decisiones
    plt.figure(figsize=(12, 8))
    plot_tree(
        clf,
        feature_names=X.columns.astype(str),  # Convertir nombres de columnas a cadenas
        class_names=[str(cls) for cls in label_encoders[dataset.columns[-1]].classes_],  # Convertir clases a cadenas
        filled=True,
        rounded=True
    )
    plt.title(f"Árbol de Decisiones (Nivel: {str(nivel) if nivel else 'Completo'})")  # Convertir nivel a cadena
    plt.savefig(f"{nombre_archivo}.png")  # Guardar el gráfico como imagen
    plt.close()
    print(f"\nÁrbol guardado como imagen: {nombre_archivo}.png")
