# arbol_decision.py
import pandas as pd
import numpy as np
from math import log2

def entropia(datos):
    clases = datos['Resultado'].value_counts()
    total = len(datos)
    return -sum((count/total) * log2(count/total) for count in clases)

def id3(datos, atributos, profundidad=0):
    clases = datos['Resultado'].unique()

    nodo = {}
    nodo['entropía'] = round(entropia(datos), 3)

    if len(clases) == 1:
        nodo['clase'] = clases[0]
        return nodo

    if not atributos:
        nodo['clase'] = datos['Resultado'].mode()[0]
        return nodo

    # Buscar mejor atributo
    mejor_atributo = None
    mejor_ganancia = -1
    entropia_total = entropia(datos)

    for atributo in atributos:
        entropia_condicional = 0
        for valor in datos[atributo].unique():
            subconjunto = datos[datos[atributo] == valor]
            peso = len(subconjunto) / len(datos)
            entropia_condicional += peso * entropia(subconjunto)
        
        ganancia = entropia_total - entropia_condicional
        if ganancia > mejor_ganancia:
            mejor_ganancia = ganancia
            mejor_atributo = atributo

    if mejor_atributo is None:
        nodo['clase'] = datos['Resultado'].mode()[0]
        return nodo

    nodo['atributo'] = mejor_atributo
    nodo['hijos'] = {}

    for valor in datos[mejor_atributo].unique():
        subconjunto = datos[datos[mejor_atributo] == valor]
        if subconjunto.empty:
            nodo['hijos'][valor] = {'clase': datos['Resultado'].mode()[0]}
        else:
            nuevo_dataset = subconjunto.drop(columns=[mejor_atributo])
            nodo['hijos'][valor] = id3(nuevo_dataset, [a for a in atributos if a != mejor_atributo], profundidad+1)

    return nodo
