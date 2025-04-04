import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter

# Leer el archivo CSV completo
df = pd.read_csv("/Users/juanpablocabreraquiroga/Documents/100-days-of-code/Day 9/questions.csv")

# Asegurarnos que question1 y question2 no tengan valores nulos
df['question1'] = df['question1'].fillna('')
df['question2'] = df['question2'].fillna('')

# Usando CountVectorizer para crear los vectores BOW (como en el notebook de referencia)
vectorizer = CountVectorizer()
q1_vector = vectorizer.fit_transform(df['question1'])
q2_vector = vectorizer.transform(df['question2'])

# Calculando la similitud de coseno entre las preguntas BOW
cosine_similarities = cosine_similarity(q1_vector, q2_vector)

# Añadiendo los resultados al dataframe
df['cos_BOW'] = cosine_similarities.diagonal()
df['q1_vector_BOW'] = list(q1_vector)
df['q2_vector_BOW'] = list(q2_vector)

# Ahora implementaremos TF-IDF de forma similar al notebook de referencia

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
        # TF: En el notebook de referencia usan count/2 como tf
        tf = count / 2
        # IDF: logaritmo del número total de documentos dividido por 
        # el número de documentos que contienen la palabra
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
df['q1_vector_TFIDF'] = df['question1'].apply(compute_tfidf)
df['q2_vector_TFIDF'] = df['question2'].apply(compute_tfidf)

# Calcular similitud usando TF-IDF
df['cos_TFIDF'] = df.apply(
    lambda row: cosine_similarity_manual(row['q1_vector_TFIDF'], row['q2_vector_TFIDF']), 
    axis=1
)

# Guardar el resultado en un nuevo CSV
output_path = "/Users/juanpablocabreraquiroga/Documents/100-days-of-code/Day 9/questions_with_similarities.csv"
df.to_csv(output_path, index=False)

print(f"Archivo guardado con éxito en: {output_path}")
print("\nMuestra de los primeros registros:")
print(df[['id', 'question1', 'question2', 'is_duplicate', 'cos_BOW', 'cos_TFIDF']].head())

# Estructura final del DataFrame:
# id, qid1, qid2, question1, question2, is_duplicate, 
# cos_BOW, q1_vector_BOW, q2_vector_BOW,
# q1_vector_TFIDF, q2_vector_TFIDF, cos_TFIDF