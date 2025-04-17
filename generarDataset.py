import pandas as pd
import random
#Este script genera un conjunto de datos sintético con atributos y etiquetas de clase aleatorios.

#Genera un conjunto de datos con valores aleatorios para color, tamaño y etiquetas de clase.
def generar_dataset(num_instancias=15):
    
    atributos_1 = ['Rojo', 'Verde', 'Azul', 'Amarillo', 'Naranja', 'Morado', 'Rosa', 'Marrón', 'Negro', 'Blanco', 'Gris', 'Cian', 'Turquesa', 'Magenta', 'Lima']
    atributos_2 = ['Pequeño', 'Mediano', 'Grande', 'Muy pequeño', 'Muy grande', 'Enorme', 'Minúsculo', 'Gigante']
    clases = ['Sí', 'No']

    data = {
        'Color': [random.choice(atributos_1) for _ in range(num_instancias)],
        'Altura': [random.choice(atributos_2) for _ in range(num_instancias)],
        'Resultado': [random.choice(clases) for _ in range(num_instancias)]
    }

    print("Dataset antes de sklearn:")
    print(pd.DataFrame(data))

    return pd.DataFrame(data)
