
# Reporte de Comparación de Técnicas de Vectorización

## Resumen de Resultados

- **Total de comparaciones realizadas:** 10
- **Bag of Words (BOW):** 4/10 aciertos (40.00%)
- **TF-IDF:** 5/10 aciertos (50.00%)
- **Cadenas de Markov:** 3/10 aciertos (30.00%)

## Resultados Detallados por Técnica y Categoría de Similitud

### Categoría: HIGH
- **Número de textos:** 4
- **BOW:** 1/4 aciertos (25.00%)
- **TF-IDF:** 4/4 aciertos (100.00%)
- **Markov:** 0/4 aciertos (0.00%)

### Categoría: MEDIUM
- **Número de textos:** 3
- **BOW:** 3/3 aciertos (100.00%)
- **TF-IDF:** 1/3 aciertos (33.33%)
- **Markov:** 0/3 aciertos (0.00%)

### Categoría: LOW
- **Número de textos:** 3
- **BOW:** 0/3 aciertos (0.00%)
- **TF-IDF:** 0/3 aciertos (0.00%)
- **Markov:** 3/3 aciertos (100.00%)

## Conclusiones

Basado en los resultados obtenidos, se puede concluir que:

1. La técnica con mejor desempeño general fue **TF-IDF**.
2. Para textos con alta similitud, TF-IDF mostró el mejor rendimiento.
3. Para textos con similitud media, BOW fue más efectivo.
4. Para textos con baja similitud, Cadenas de Markov logró mejores resultados.

Estas observaciones sugieren que cada técnica tiene sus propias fortalezas y debilidades dependiendo del nivel de similitud entre los textos. Para aplicaciones prácticas, podría ser beneficioso implementar un enfoque híbrido que combine estas técnicas según el contexto específico.
