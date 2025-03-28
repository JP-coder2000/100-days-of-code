import random
import pandas as pd
import matplotlib.pyplot as plt

seis = [1,0]
dado = [1,2,3,4,5]
tiradas = 1000
resultados = []


for i in range(tiradas):
    if random.choice(seis) == 1:
        resultados.append(6)
    else:
        resultados.append(random.choice(dado)) 
        
plt.hist(resultados, bins=11, align='mid', color='blue', edgecolor='black')
plt.show()