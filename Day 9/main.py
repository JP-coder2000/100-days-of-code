import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from collections import Counter
import re
import string
import os
import glob

# Función de preprocesamiento para el texto
def preprocess_text(text):
    if not isinstance(text, str):
        return []
    text = text.lower()
    text = re.sub(f'[{string.punctuation}]', '', text)
    return text.split()

# Función para leer los archivos de texto
def read_text_file(file_path):
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        print(f"Error al leer el archivo {file_path}: {e}")
        return ""

# 1. Vectorización con BOW
def calculate_bow_similarity(text1, text2):
    vectorizer = CountVectorizer()
    vectors = vectorizer.fit_transform([text1, text2])
    return cosine_similarity(vectors)[0, 1]

# 2. Vectorización con TF-IDF
def calculate_tfidf_similarity(text1, text2):
    # Preprocesar textos
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)
    
    # Juntar todos los textos para calcular IDF
    all_words = set(words1 + words2)
    
    # Calcular document frequency
    document_count = {}
    for word in set(words1):
        document_count[word] = document_count.get(word, 0) + 1
    for word in set(words2):
        document_count[word] = document_count.get(word, 0) + 1
    
    # Función para calcular TF-IDF de un texto
    def compute_tfidf_vector(words):
        word_count = Counter(words)
        total_words = len(words) if len(words) > 0 else 1
        tfidf_vector = {}
        
        for word in all_words:
            # TF: Term Frequency
            tf = word_count.get(word, 0) / total_words
            # IDF: Inverse Document Frequency (2 documentos en total)
            idf = np.log(2 / (document_count.get(word, 0) + 1))
            tfidf_vector[word] = tf * idf
            
        return tfidf_vector
    
    # Calcular vectores TF-IDF
    vec1 = compute_tfidf_vector(words1)
    vec2 = compute_tfidf_vector(words2)
    
    # Calcular similitud del coseno
    all_words_list = list(all_words)
    q1_vector = np.array([vec1.get(word, 0.0) for word in all_words_list])
    q2_vector = np.array([vec2.get(word, 0.0) for word in all_words_list])
    
    # Calcular similitud del coseno
    dot_product = np.dot(q1_vector, q2_vector)
    norm_q1 = np.linalg.norm(q1_vector)
    norm_q2 = np.linalg.norm(q2_vector)
    
    return dot_product / (norm_q1 * norm_q2) if norm_q1 * norm_q2 != 0 else 0.0

# 3. Vectorización con Cadenas de Markov
def calculate_markov_similarity(text1, text2):
    words1 = preprocess_text(text1)
    words2 = preprocess_text(text2)
    
    # Si alguna pregunta está vacía o tiene solo una palabra, devuelve 0
    if len(words1) <= 1 or len(words2) <= 1:
        return 0.0
    
    # Crear el vocabulario único
    unique_words = list(set(words1 + words2))
    word_to_index = {word: i for i, word in enumerate(unique_words)}
    
    # Matrices de transición
    n = len(unique_words)
    matrix1 = np.zeros((n, n))
    matrix2 = np.zeros((n, n))
    
    # Construir matriz para texto1
    for i in range(len(words1) - 1):
        curr_word = words1[i]
        next_word = words1[i + 1]
        curr_idx = word_to_index[curr_word]
        next_idx = word_to_index[next_word]
        matrix1[curr_idx, next_idx] += 1
    
    # Construir matriz para texto2
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
    
    # Calcular similitud del coseno
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 * norm_vec2 != 0 else 0.0

# Función para determinar el nivel de similitud según los intervalos dados
def get_expected_similarity(file_name):
    if "high" in file_name.lower():
        return "high"
    elif "moderate" in file_name.lower():
        return "medium"
    elif "low" in file_name.lower():
        return "low"
    else:
        return "unknown"

# Función para verificar si la técnica detectó correctamente la similitud
def is_correct_detection(similarity_score, expected_category):
    if expected_category == "high":
        return 0.85 <= similarity_score <= 1.00
    elif expected_category == "medium":
        return 0.45 <= similarity_score < 0.85
    elif expected_category == "low":
        return 0.0 <= similarity_score < 0.45
    else:
        return False

def main():
    # Directorio donde se encuentran los archivos
    directory = "./texts"  # Ajusta esta ruta según tu estructura
    
    # Archivo original
    original_file = os.path.join(directory, "original.txt")
    original_text = read_text_file(original_file)
    
    # Obtener todos los archivos similares
    similar_files = []
    for pattern in ["high_*.txt", "moderate_*.txt", "low_*.txt"]:
        similar_files.extend(glob.glob(os.path.join(directory, pattern)))
    
    # Crear una lista para almacenar los resultados
    results = []
    
    # Contadores para el reporte final
    correct_bow = 0
    correct_tfidf = 0
    correct_markov = 0
    total_comparisons = 0
    
    # Comparar el archivo original con cada archivo similar
    for similar_file in similar_files:
        similar_text = read_text_file(similar_file)
        similar_file_name = os.path.basename(similar_file)
        
        # Calcular similitudes
        bow_similarity = calculate_bow_similarity(original_text, similar_text)
        tfidf_similarity = calculate_tfidf_similarity(original_text, similar_text)
        markov_similarity = calculate_markov_similarity(original_text, similar_text)
        
        # Determinar la categoría esperada
        expected_category = get_expected_similarity(similar_file_name)
        
        # Verificar si cada técnica detectó correctamente la similitud
        bow_correct = is_correct_detection(bow_similarity, expected_category)
        tfidf_correct = is_correct_detection(tfidf_similarity, expected_category)
        markov_correct = is_correct_detection(markov_similarity, expected_category)
        
        # Actualizar contadores
        if bow_correct:
            correct_bow += 1
        if tfidf_correct:
            correct_tfidf += 1
        if markov_correct:
            correct_markov += 1
        total_comparisons += 1
        
        # Agregar resultado a la lista
        results.append({
            "Archivo_Original": "original.txt",
            "Archivo_Similar": similar_file_name,
            "Nivel_Similitud": expected_category,
            "Coseno_BOW": bow_similarity,
            "BOW_Correcto": bow_correct,
            "Coseno_TFIDF": tfidf_similarity,
            "TFIDF_Correcto": tfidf_correct,
            "Coseno_Markov": markov_similarity,
            "Markov_Correcto": markov_correct
        })
    
    # Crear DataFrame con los resultados
    df_results = pd.DataFrame(results)
    
    # Guardar resultados en un archivo CSV
    csv_output = "resultados_similitud.csv"
    df_results.to_csv(csv_output, index=False)
    
    # Generar un informe simple
    print(f"Archivo CSV generado: {csv_output}")
    print("\nResultados de la comparación:")
    print(f"Total de comparaciones realizadas: {total_comparisons}")
    print(f"BOW acertó correctamente: {correct_bow}/{total_comparisons} ({correct_bow/total_comparisons*100:.2f}%)")
    print(f"TF-IDF acertó correctamente: {correct_tfidf}/{total_comparisons} ({correct_tfidf/total_comparisons*100:.2f}%)")
    print(f"Markov acertó correctamente: {correct_markov}/{total_comparisons} ({correct_markov/total_comparisons*100:.2f}%)")
    
    # Crear un informe más detallado
    report = f"""
# Reporte de Comparación de Técnicas de Vectorización

## Resumen de Resultados

- **Total de comparaciones realizadas:** {total_comparisons}
- **Bag of Words (BOW):** {correct_bow}/{total_comparisons} aciertos ({correct_bow/total_comparisons*100:.2f}%)
- **TF-IDF:** {correct_tfidf}/{total_comparisons} aciertos ({correct_tfidf/total_comparisons*100:.2f}%)
- **Cadenas de Markov:** {correct_markov}/{total_comparisons} aciertos ({correct_markov/total_comparisons*100:.2f}%)

## Resultados Detallados por Técnica y Categoría de Similitud
"""
    
    # Agregar resultados por categoría
    for category in ["high", "medium", "low"]:
        category_df = df_results[df_results["Nivel_Similitud"] == category]
        cat_total = len(category_df)
        
        if cat_total > 0:
            bow_cat = sum(category_df["BOW_Correcto"])
            tfidf_cat = sum(category_df["TFIDF_Correcto"])
            markov_cat = sum(category_df["Markov_Correcto"])
            
            report += f"""
### Categoría: {category.upper()}
- **Número de textos:** {cat_total}
- **BOW:** {bow_cat}/{cat_total} aciertos ({bow_cat/cat_total*100:.2f}%)
- **TF-IDF:** {tfidf_cat}/{cat_total} aciertos ({tfidf_cat/cat_total*100:.2f}%)
- **Markov:** {markov_cat}/{cat_total} aciertos ({markov_cat/cat_total*100:.2f}%)
"""
    
    # Añadir conclusiones
    best_method = max([(correct_bow, "BOW"), (correct_tfidf, "TF-IDF"), (correct_markov, "Cadenas de Markov")])[1]
    
    report += f"""
## Conclusiones

Basado en los resultados obtenidos, se puede concluir que:

1. La técnica con mejor desempeño general fue **{best_method}**.
2. Para textos con alta similitud, {("BOW" if sum(df_results[df_results["Nivel_Similitud"]=="high"]["BOW_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="high"]) >= sum(df_results[df_results["Nivel_Similitud"]=="high"]["TFIDF_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="high"]) and sum(df_results[df_results["Nivel_Similitud"]=="high"]["BOW_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="high"]) >= sum(df_results[df_results["Nivel_Similitud"]=="high"]["Markov_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="high"]) else "TF-IDF" if sum(df_results[df_results["Nivel_Similitud"]=="high"]["TFIDF_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="high"]) >= sum(df_results[df_results["Nivel_Similitud"]=="high"]["Markov_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="high"]) else "Cadenas de Markov")} mostró el mejor rendimiento.
3. Para textos con similitud media, {("BOW" if sum(df_results[df_results["Nivel_Similitud"]=="medium"]["BOW_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="medium"]) >= sum(df_results[df_results["Nivel_Similitud"]=="medium"]["TFIDF_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="medium"]) and sum(df_results[df_results["Nivel_Similitud"]=="medium"]["BOW_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="medium"]) >= sum(df_results[df_results["Nivel_Similitud"]=="medium"]["Markov_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="medium"]) else "TF-IDF" if sum(df_results[df_results["Nivel_Similitud"]=="medium"]["TFIDF_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="medium"]) >= sum(df_results[df_results["Nivel_Similitud"]=="medium"]["Markov_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="medium"]) else "Cadenas de Markov")} fue más efectivo.
4. Para textos con baja similitud, {("BOW" if sum(df_results[df_results["Nivel_Similitud"]=="low"]["BOW_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="low"]) >= sum(df_results[df_results["Nivel_Similitud"]=="low"]["TFIDF_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="low"]) and sum(df_results[df_results["Nivel_Similitud"]=="low"]["BOW_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="low"]) >= sum(df_results[df_results["Nivel_Similitud"]=="low"]["Markov_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="low"]) else "TF-IDF" if sum(df_results[df_results["Nivel_Similitud"]=="low"]["TFIDF_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="low"]) >= sum(df_results[df_results["Nivel_Similitud"]=="low"]["Markov_Correcto"])/len(df_results[df_results["Nivel_Similitud"]=="low"]) else "Cadenas de Markov")} logró mejores resultados.

Estas observaciones sugieren que cada técnica tiene sus propias fortalezas y debilidades dependiendo del nivel de similitud entre los textos. Para aplicaciones prácticas, podría ser beneficioso implementar un enfoque híbrido que combine estas técnicas según el contexto específico.
"""
    
    # Guardar el reporte en un archivo
    report_file = "reporte_similitud.md"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReporte detallado generado: {report_file}")

if __name__ == "__main__":
    main()