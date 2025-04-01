import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

# Leer el archivo CSV completo, para mantener todas las columnas originales
df = pd.read_csv("/Users/juanpablocabreraquiroga/Documents/100-days-of-code/Day 9/questions.csv")

# Asegurarnos que question1 y question2 no tengan valores nulos
df['question1'] = df['question1'].fillna('')
df['question2'] = df['question2'].fillna('')

# Función para tokenizar el texto (dividir en palabras)
def tokenize(text):
    # Podemos mejorar esta función para manejar puntuación, mayúsculas, etc.
    return text.lower().split()

# Procesar cada fila para crear vocabulario y vectores
def calculate_similarity(row):
    # Tokenizar ambas preguntas
    tokens1 = tokenize(row['question1'])
    tokens2 = tokenize(row['question2'])
    
    # Crear vocabulario único (todas las palabras distintas)
    vocabulary = sorted(list(set(tokens1 + tokens2)))
    
    # Crear vectores para cada pregunta
    vector1 = []
    vector2 = []
    
    # Calcular frecuencia de cada palabra del vocabulario en cada pregunta
    for word in vocabulary:
        vector1.append(tokens1.count(word))
        vector2.append(tokens2.count(word))
    
    # Calcular similitud de coseno
    if sum(vector1) > 0 and sum(vector2) > 0:  # Evitar división por cero
        similarity = cosine_similarity([vector1], [vector2])[0][0]
    else:
        similarity = 0
    
    return similarity

# Aplicar la función a cada fila para calcular solo la similitud
df['similarity'] = df.apply(calculate_similarity, axis=1)

# Guardar el resultado en un nuevo CSV
output_path = "/Users/juanpablocabreraquiroga/Documents/100-days-of-code/Day 9/questions_with_similarity.csv"
df.to_csv(output_path, index=False)

print(f"Archivo guardado con éxito en: {output_path}")
print(f"Muestra de los primeros registros con similitud:\n{df[['question1', 'question2', 'similarity']].head()}")