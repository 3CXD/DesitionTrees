import pandas as pd
import numpy as np
from math import log2

# Función para calcular la entropía de un conjunto de datos
def entropia(datos):
    # Contar la frecuencia de cada clase en la columna 'Resultado'
    clases = datos['Resultado'].value_counts()
    total = len(datos)
    # Calcular la entropía usando la fórmula: -Σ(p * log2(p))
    return -sum((count/total) * log2(count/total) for count in clases)

# Implementación del algoritmo ID3 para construir un árbol de decisión
def id3(datos, atributos, profundidad=0):
    # Obtener las clases únicas en la columna 'Resultado'
    clases = datos['Resultado'].unique()

    # Crear un nodo para el árbol
    nodo = {}
    nodo['entropía'] = round(entropia(datos), 3)  # Calcular y almacenar la entropía del nodo actual

    # Caso base 1: Si todas las instancias pertenecen a la misma clase
    if len(clases) == 1:
        nodo['clase'] = clases[0]  # Asignar la clase al nodo
        return nodo

    # Caso base 2: Si no hay más atributos para dividir
    if not atributos:
        nodo['clase'] = datos['Resultado'].mode()[0]  # Asignar la clase más frecuente
        return nodo

    # Buscar el mejor atributo para dividir
    mejor_atributo = None
    mejor_ganancia = -1
    entropia_total = entropia(datos)  # Entropía del nodo actual

    # Evaluar cada atributo
    for atributo in atributos:
        entropia_condicional = 0
        # Calcular la entropía condicional para cada valor del atributo
        for valor in datos[atributo].unique():
            subconjunto = datos[datos[atributo] == valor]  # Filtrar subconjunto
            peso = len(subconjunto) / len(datos)  # Peso del subconjunto
            entropia_condicional += peso * entropia(subconjunto)  # Sumar contribución ponderada

        # Calcular la ganancia de información
        ganancia = entropia_total - entropia_condicional
        # Actualizar el mejor atributo si la ganancia es mayor
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_atributo = atributo

    # Caso base 3: Si no se encuentra un atributo para dividir
    if mejor_atributo is None:
        nodo['clase'] = datos['Resultado'].mode()[0]  # Asignar la clase más frecuente
        return nodo

    # Dividir el nodo usando el mejor atributo
    nodo['atributo'] = mejor_atributo
    nodo['hijos'] = {}

    # Crear nodos hijos para cada valor del mejor atributo
    for valor in datos[mejor_atributo].unique():
        subconjunto = datos[datos[mejor_atributo] == valor]  # Filtrar subconjunto
        if subconjunto.empty:
            # Si el subconjunto está vacío, asignar la clase más frecuente
            nodo['hijos'][valor] = {'clase': datos['Resultado'].mode()[0]}
        else:
            # Llamada recursiva para construir el árbol en el subconjunto
            nuevo_dataset = subconjunto.drop(columns=[mejor_atributo])  # Eliminar el atributo usado
            nodo['hijos'][valor] = id3(nuevo_dataset, [a for a in atributos if a != mejor_atributo], profundidad+1)

    return nodo
