import matplotlib.pyplot as plt
import numpy as np
import math

data_path = "/Users/juanpablocabreraquiroga/Documents/100-days-of-code/Day 7/data03.txt"

# Leer y ordenar los datos
with open(data_path, "r") as file:
    lines = file.readlines()
    data = [float(line.strip()) for line in lines]
    data.sort()

# Cálculo del número de clases usando la regla de Sturges
N = len(data)
C = math.ceil(1 + (3.3 * np.log10(N)))

# Cálculo del rango y ancho de clase
min_value = min(data)
max_value = max(data)
rango = max_value - min_value
W = rango / C  


# Creación de los límites de los intervalos
bin_edges = []
for i in range(C + 1):
    bin_edges.append(min_value + i * W)

# Cálculo manual de frecuencias para cada intervalo
frequencies = [0] * C
for value in data:
    for i in range(C):
        if bin_edges[i] <= value < bin_edges[i+1] or (i == C-1 and value == bin_edges[i+1]):
            frequencies[i] += 1
            break

interval_labels = []
for i in range(C):
    if i == C-1:
        interval_labels.append(f"[{bin_edges[i]:.4f} - {bin_edges[i+1]:.4f}]")
    else:
        interval_labels.append(f"[{bin_edges[i]:.4f} - {bin_edges[i+1]:.4f})")

# Impresión de información
print(f"Número de datos (N): {N}")
print(f"Número de clases (C): {C} (Sturges: 1 + 3.3 × log10({N}))")
print(f"Valor mínimo: {min_value:.4f}")
print(f"Valor máximo: {max_value:.4f}")
print(f"Rango: {rango:.4f}")
print(f"Ancho de clase (W): {W:.4f}")

print(f"\nSuma total de frecuencias: {sum(frequencies)} (debe ser igual a N: {N})")


print("\nIntervalos y Frecuencias:")
for i in range(C):
    print(f"{interval_labels[i]}: {frequencies[i]}")

# Creación del histograma
plt.figure(figsize=(12, 6))

plt.hist(data, bins=bin_edges, color='#3274A1', edgecolor='black')

plt.xlabel('Bins')
plt.ylabel('Frequencies')
plt.title('Frequencies of grouped data')

bin_centers = [min_value + (i + 0.5) * W for i in range(C)]
plt.xticks(bin_centers, interval_labels, rotation=45, ha='right')

plt.tight_layout()

# Mostrar el gráfico
plt.show()

