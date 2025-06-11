
#Imprimir números del 0 al 9
"""x = range(10)
for i in x:
    print(i)
#Imprimir números del 5 al 15
# ¿Qué pasa si haces esto?
numeros = list(range(5,16))
print(numeros)  # ¿Qué crees que sale?

#Imprimir números pares del 0 al 20 (0, 2, 4, 6...)
x = range(0,22,2)
for i in x:
    print(i)

numeros = list(range(0,22,2))
print(numeros)  # ¿Qué crees que sale?

#Dada una lista, imprimir cuántos elementos tiene
pythonlista = ['perro', 'gato', 'pez', 'loro']
x = len(pythonlista)
print(x)
# Resultado esperado: 4

#5. Imprimir cada elemento de una lista junto con su posición
pythoncolores = ['rojo', 'azul', 'verde']
# Resultado esperado:
# 0: rojo
# 1: azul  
# 2: verde
# Nota personal, el enumerate primero te da el indice y despues el valor de ese indice
for posicion, valor in enumerate(pythoncolores):
    print(posicion,valor)
    
    
#6. Crear una lista con números del 10 al 20
#(Pista: No uses for, usa algo que convierta el generador directamente en lista)

nums = list(range(10,21))
print(numeros)

#Recorrer una lista e imprimir solo los elementos en posiciones pares (0, 2, 4...)
pythonfrutas = ['manzana', 'banana', 'uva', 'pera', 'kiwi']
# Resultado: manzana, uva, kiwi

for pos,val in enumerate(pythonfrutas):
    if pos % 2 ==0:
        print(val)

#8. Encontrar el elemento más largo de una lista
palabras = ['casa', 'elefante', 'sol', 'programación']
# Resultado esperado: 'programación'

elemento_mas_largo = ""
for pos,val in enumerate(palabras):
    if elemento_mas_largo == "":
            elemento_mas_largo = val
    else:
        if len(val) > len(elemento_mas_largo):
            elemento_mas_largo = val
            
            
"""
#9. Crear dos listas separadas: una con números pares y otra con impares
pythonnumeros = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
pares = []
impares = []
# Resultado: pares = [2, 4, 6, 8, 10], impares = [1, 3, 5, 7, 9]
for pos,val in enumerate(pythonnumeros):
    if val % 2 == 0:
        pares.append(val)
    else:
        impares.append(val)

#print(pares)
#print(impares)


#10. Imprimir elementos de dos listas en paralelo
pythonnombres = ['Ana', 'Luis', 'María']
edades = [25, 30, 28]
# Resultado esperado: 
# Ana tiene 25 años
# Luis tiene 30 años  
# María tiene 28 años

#for pos_nombres,val_nombres in enumerate(pythonnombres):
    #print(f"{val_nombres} tiene {edades[pos_nombres]} años")
    
    
#
# 
# 11. Sumar elementos de dos listas en paralelo
pythonlista1 = [1, 3, 5, 7]
lista2 = [2, 4, 6, 8]
# Resultado: [3, 7, 11, 15]
resultado = []
# Resolviendo con zip
for l1, l2 in zip(pythonlista1,lista2):
    resultado.append(l1+l2)
    
    #Lo intente poner como print(list(l1+l2)), pero no me salio.
#print(resultado)
#Resolviendo sin zip
#for pos,val in enumerate(pythonlista1):
    #print(val+lista2[pos])
    
    
frase = "Python es increíble y poderoso"
palabras = frase.split()
#print(palabras)
#print(len(palabras))

#13. Convertir una lista de palabras en una sola frase
pythonpalabras = ["Mañana", "tengo", "mi", "entrevista"]
# Resultado: "Mañana tengo mi entrevista"

#junta =" ".join(pythonpalabras)
#print(junta)


#14. Ordenar una lista de números SIN modificar la original
pythonnumeros = [5, 2, 8, 1, 9]
# numeros debe seguir siendo [5, 2, 8, 1, 9]
# Pero necesitas una nueva lista ordenada: [1, 2, 5, 8, 9]

res = sorted(pythonnumeros)
#print(res)



#15. Crear un diccionario combinando dos listas
pythonkeys = ['nombre', 'edad', 'ciudad']
values = ['Carlos', 28, 'Madrid']
# Resultado: {'nombre': 'Carlos', 'edad': 28, 'ciudad': 'Madrid'}

#Lo voy  ahacer de las dos maneras, con zip y append, aunque creo que no existe append para diccionarios, solo listas...

res = dict(zip(pythonkeys,values))

#print(res)


#16. Encontrar el elemento que más se repite en una lista
pythonnumeros = [3, 7, 3, 3, 5, 7, 1, 3, 5]
# Resultado: 3 (aparece 4 veces)


conteo = {}
elementoRepetido =[]
for pos,val in enumerate(pythonnumeros):
    if val in conteo:
        conteo[val] = conteo[val] +1
        # esto quiere decir que del diccionario conteo, en la clave de val, voy a aumentar 1 en su valor?
        # 
    else:
        conteo[val] = 1
        # aqui quise hacer el, si no existe, pues agrega en ese indice un 1
        

numero_mas_repetido = 0
max_repeticiones = 0

for numero, repeticiones in conteo.items():
    if repeticiones > max_repeticiones:
        max_repeticiones = repeticiones
        numero_mas_repetido = numero

#print(numero_mas_repetido)


#17. Eliminar elementos duplicados de una lista (mantener orden)
pythonlista = [1, 2, 3, 2, 4, 1, 5, 3]
# Resultado: [1, 2, 3, 4, 5]

# me voy a ver super inteligente jajaja
a = set(pythonlista)
#print(a)

# Ahora, si quiero regresar una lista como tal:

conocidos = []
vistos = set()

for val in pythonlista:
    if val not in vistos:  # ¿Ya lo vimos antes?
        conocidos.append(val)  # Solo si es nuevo
        vistos.add(val)
        
#print(conocidos)

# palabra es un palindromo
palabra = "reconocer"
# Resultado: True (se lee igual al revés)

#Lo estoy pensando en hacer de varias maneras.
#  metodo de slicing 
a = list(palabra)
#print(a)

x = a[::-1]
string = "".join(x)
#print(string)
#if palabra == string:
    #print("True")
#else:
    #print("False")

palabra = "reconocer"
a = list(palabra)
es_palindromo = True  # Asumimos que sí es

for pos, val in enumerate(a):
    # Comparar a[pos] con su posición simétrica
    if a[pos] != a[-(pos + 1)]:
        # en la primera itreacion revisa el indice 0 de a contra a- (pos es 0+1) 1
        # y conforme a pasando se va aumentando el contador de pos... woooow
        es_palindromo = False
        break  # Ya encontramos una diferencia
        
#if es_palindromo:
#    print("True")
#else:
#    print("False")
    
numeros = [1, 8, 3, 6, 2, 10, 4, 9]
# Resultado: [8, 6, 10]

condiciones =[]
for i in numeros:
    if i % 2 ==0 and i>5:
        condiciones.append(i)
#print(condiciones)

# Contar cuantas veces aparace una letra en una palabra
palabra = "programacion"
# Resultado: {'p': 1, 'r': 3, 'o': 2, 'g': 1, 'a': 3, 'm': 1, 'c': 1, 'i': 1, 'n': 1}

x = list(palabra)
contador = {}
for pos,val in enumerate(x):
    if val not in contador:
        contador[val] = 1
    else:
        contador[val] = contador[val] +1

#print(contador)

letra_mas_repetida = ""
numero_de_reps = 0

for letra,repe in contador.items():
    if repe > numero_de_reps:
        letra_mas_repetida = letra
        numero_de_reps = repe

#print(letra_mas_repetida)


numeros = [1, 8, 3, 6, 2, 10, 4, 9]
acomodados = []
# [1,8]

for pos,val in enumerate(numeros):
    insertado = False
    if len(acomodados) == 0:
        acomodados.append(val)
    else:
        for pos_acomodados, val_acomodados in enumerate(acomodados):
            if val > val_acomodados:
                acomodados.insert(pos_acomodados,val)
                insertado = True
                break
        if insertado == False:
            acomodados.append(val)
            
print(acomodados)