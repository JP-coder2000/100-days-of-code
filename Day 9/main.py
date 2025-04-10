import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import string
import json

# Leer el archivo CSV completo
df = pd.read_csv("/Users/juanpablocabreraquiroga/Documents/100-days-of-code/Day 9/questions.csv")

# Asegurarnos que question1 y question2 no tengan valores nulos
df['question1'] = df['question1'].fillna('')
df['question2'] = df['question2'].fillna('')

# Usando CountVectorizer para crear los vectores BOW
vectorizer = CountVectorizer()
q1_vector = vectorizer.fit_transform(df['question1'])
q2_vector = vectorizer.transform(df['question2'])

# Calculando la similitud de coseno entre las preguntas BOW
cosine_similarities = cosine_similarity(q1_vector, q2_vector)

# Añadiendo los resultados al dataframe
df['cos_BOW'] = cosine_similarities.diagonal()

# Convertir matrices dispersas a listas serializables
def convert_sparse_to_list(sparse_matrix):
    # Convertir a array denso y luego a lista
    return sparse_matrix.toarray()[0].tolist()

# Guardar vectores BOW como listas normales en lugar de matrices dispersas
df['q1_vecBOW'] = [convert_sparse_to_list(q1_vector[i]) for i in range(q1_vector.shape[0])]
df['q2_vecBOW'] = [convert_sparse_to_list(q2_vector[i]) for i in range(q2_vector.shape[0])]

# Recopilamos todas las preguntas como documentos
all_questions = pd.concat([df['question1'], df['question2']]).reset_index(drop=True)

# Calculamos el document count (número de documentos que contienen cada palabra)
document_count = {}
total_documents = len(all_questions)

for text in all_questions:
    words = set(text.lower().split())
    for word in words:
        document_count[word] = document_count.get(word, 0) + 1

# Función para calcular los vectores TF-IDF para un texto
def compute_tfidf(text):
    words = text.lower().split()
    word_count = Counter(words)
    total_words = len(words) if len(words) > 0 else 1
    tfidf_vector = {}

    for word, count in word_count.items():
        # TF: Term Frequency
        tf = count / total_words
        # IDF: Inverse Document Frequency
        idf = np.log(total_documents / (document_count.get(word, 0) + 1))
        tfidf_vector[word] = tf * idf

    return tfidf_vector

# Función para calcular similitud de coseno entre vectores TF-IDF
def cosine_similarity_manual(vec1, vec2):
    all_words = set(vec1.keys()).union(set(vec2.keys()))

    # Crear vectores con los valores TF-IDF
    q1_vector = np.array([vec1.get(word, 0.0) for word in all_words])
    q2_vector = np.array([vec2.get(word, 0.0) for word in all_words])

    # Calcular similitud del coseno
    dot_product = np.dot(q1_vector, q2_vector)
    norm_q1 = np.linalg.norm(q1_vector)
    norm_q2 = np.linalg.norm(q2_vector)

    return dot_product / (norm_q1 * norm_q2) if norm_q1 * norm_q2 != 0 else 0.0

# Calcular vectores TF-IDF para cada pregunta
df['q1_vecTFIDF'] = df['question1'].apply(compute_tfidf)
df['q2_vecTFIDF'] = df['question2'].apply(compute_tfidf)

# Calcular similitud usando TF-IDF
df['cos_TFIDF'] = df.apply(
    lambda row: cosine_similarity_manual(row['q1_vecTFIDF'], row['q2_vecTFIDF']), 
    axis=1
)

# Función de preprocesamiento para Markov
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    return text.split()

# Implementar Markov Chain
def compute_markov_chain(question1, question2):
    words1 = preprocess_text(question1)
    words2 = preprocess_text(question2)
    
    # Si alguna pregunta está vacía o tiene solo una palabra, devuelve vectores vacíos
    if len(words1) <= 1 and len(words2) <= 1:
        return np.array([0]), np.array([0])
    
    # Crear el vocabulario único
    unique_words = list(set(words1 + words2))
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    
    # Matrices de transición
    n = len(unique_words)
    matrix1 = np.zeros((n, n))
    matrix2 = np.zeros((n, n))
    
    # Construir matriz para question1
    for i in range(len(words1) - 1):
        curr_word = words1[i]
        next_word = words1[i + 1]
        curr_idx = word_to_index[curr_word]
        next_idx = word_to_index[next_word]
        matrix1[curr_idx, next_idx] += 1
    
    # Construir matriz para question2
    for i in range(len(words2) - 1):
        curr_word = words2[i]
        next_word = words2[i + 1]
        curr_idx = word_to_index[curr_word]
        next_idx = word_to_index[next_word]
        matrix2[curr_idx, next_idx] += 1
    
    # Normalizar para obtener probabilidades
    for i in range(n):
        row_sum1 = np.sum(matrix1[i, :])
        row_sum2 = np.sum(matrix2[i, :])
        
        if row_sum1 > 0:
            matrix1[i, :] = matrix1[i, :] / row_sum1
        
        if row_sum2 > 0:
            matrix2[i, :] = matrix2[i, :] / row_sum2
    
    # Convertir matrices a vectores planos
    vec1 = matrix1.flatten()
    vec2 = matrix2.flatten()
    
    return vec1, vec2

# Calcular vectores de Markov para cada par de preguntas
markov_results = [compute_markov_chain(q1, q2) for q1, q2 in zip(df['question1'], df['question2'])]

# Extraer vectores
df['q1_vecMARK'] = [vec[0].tolist() for vec in markov_results]
df['q2_vecMARK'] = [vec[1].tolist() for vec in markov_results]

# Función para calcular similitud de Markov
def calculate_markov_similarity(vec1, vec2):
    # Convertir a arrays numpy
    vec1_np = np.array(vec1)
    vec2_np = np.array(vec2)
    
    # Si algún vector está vacío, la similitud es 0
    if len(vec1_np) == 0 or len(vec2_np) == 0:
        return 0.0
    
    # Calcular similitud del coseno
    dot_product = np.dot(vec1_np, vec2_np)
    norm_vec1 = np.linalg.norm(vec1_np)
    norm_vec2 = np.linalg.norm(vec2_np)
    
    if norm_vec1 == 0 or norm_vec2 == 0:
        return 0.0
    
    return dot_product / (norm_vec1 * norm_vec2)

# Calcular similitud usando Markov Chain
df['cos_MARK'] = [calculate_markov_similarity(vec1, vec2) for vec1, vec2 in zip(df['q1_vecMARK'], df['q2_vecMARK'])]

# Crear una versión limpia del DataFrame para exportar a Excel
df_excel = df[['id', 'qid1', 'qid2', 'question1', 'question2', 'is_duplicate', 
               'cos_BOW', 'cos_TFIDF', 'cos_MARK']].copy()

# Convertir los diccionarios TFIDF a cadenas JSON para mejor visualización en Excel
df_excel['q1_vecTFIDF_sample'] = df['q1_vecTFIDF'].apply(
    lambda x: json.dumps({k: round(float(v), 4) for k, v in list(x.items())[:5]})
)
df_excel['q2_vecTFIDF_sample'] = df['q2_vecTFIDF'].apply(
    lambda x: json.dumps({k: round(float(v), 4) for k, v in list(x.items())[:5]})
)

# Para vectores BOW y MARK, mostrar sólo los primeros 5 elementos
df_excel['q1_vecBOW_sample'] = df['q1_vecBOW'].apply(lambda x: str(x[:5]) + '...')
df_excel['q2_vecBOW_sample'] = df['q2_vecBOW'].apply(lambda x: str(x[:5]) + '...')
df_excel['q1_vecMARK_sample'] = df['q1_vecMARK'].apply(lambda x: str(x[:5]) + '...')
df_excel['q2_vecMARK_sample'] = df['q2_vecMARK'].apply(lambda x: str(x[:5]) + '...')

# Guardar ambas versiones: completa para análisis y simplificada para Excel
output_path = "/Users/juanpablocabreraquiroga/Documents/100-days-of-code/Day 9/questions_with_similarities.csv"
output_excel = "/Users/juanpablocabreraquiroga/Documents/100-days-of-code/Day 9/questions_excel_friendly.xlsx"

# Guardar versión completa en CSV
df.to_csv(output_path, index=False)

# Guardar versión para Excel
df_excel.to_excel(output_excel, index=False)

print(f"Archivo CSV guardado con éxito en: {output_path}")
print(f"Archivo Excel guardado con éxito en: {output_excel}")
print("\nMuestra de los primeros registros:")
print(df_excel[['id', 'question1', 'question2', 'is_duplicate', 'cos_BOW', 'cos_TFIDF', 'cos_MARK']].head())

# Limitar el procesamiento a un número específico de registros (opcional)
# Para procesar solo los primeros N registros, descomenta la siguiente línea:
# df = df.head(10)  # Procesar solo 10 registros