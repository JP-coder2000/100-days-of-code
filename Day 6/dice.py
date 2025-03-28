import random
import pandas as pd
import matplotlib.pyplot as plt

dado = [1,2,3,4,5,6]
tiradas = 1000
resultados = []

for i in range(tiradas):
    resultados.append(random.choice(dado))
    
plt.hist(resultados, bins=11, align='mid', color='blue', edgecolor='black')
plt.show()
