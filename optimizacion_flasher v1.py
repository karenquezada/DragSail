import numpy as np
from deap import base, creator, tools, algorithms
import random
from multiprocessing import Pool
from flasher_lib import *

def N(r, s_f, d_function, h, epsilon, m):
    # Implementar la lógica para calcular d(i, i+1) utilizando d_function
    sum_d = sum(d_function(i, i + 1) for i in range(r))
    return int((s_f * sum_d) / (h * np.cos(epsilon / 2) * np.sqrt(1/np.sin(np.pi / m) * 1/np.sin(np.pi / m - epsilon))))

r_i_j_cache = {}

def r_i_j(params):
    # Convertir la entrada de params en una tupla, para que pueda ser usada como clave en el diccionario
    params_tuple = tuple(params)

    # Verificar si el resultado para estos parámetros ya está en la caché
    if params_tuple in r_i_j_cache:
        #print("caché", r_i_j_cache)
        return r_i_j_cache[params_tuple]
        

    # Si no está en la caché, realizar el cálculo
    i, j, m, i_final, delta, alpha, epsilon, altura = params
    resultados = []
    distancia_k = dk(altura, epsilon, m)
    #print("distancia_k fn1", distancia_k)
    punto_inicial = p(0, j, m, delta, alpha, epsilon)
    punto_actual = punto_inicial 

    modulos_totales=lista_modulos(j, i_final, m, delta, alpha, epsilon, distancia_k)[0]
    #print("modulos_totales fn1",modulos_totales)
    total_modulos = lista_modulos(j, i_final, m, delta, alpha, epsilon, distancia_k)[2]
    #print("total_modulos fn1",total_modulos)
    #print("numero de ptos fn2",total_modulos/distancia_k)
    if total_modulos < distancia_k:
        punto_actual = punto_inicial
    else:
        indice = encontrar_indice(modulos_totales, distancia_k * i)
        if indice == -1:
            return None
        
        remanente = distancia_k * i - sum(modulos_totales[:indice])
        resultados.append([indice, remanente])

        for idx, remanente in resultados:
            if idx >= i:
                break
        p_start = p(idx, j, m, delta, alpha, epsilon)
        p_next = p(idx + 1, j, m, delta, alpha, epsilon)
        direccion = (p_next - p_start) / e_distance(p_next, p_start)
        desplazamiento = direccion * remanente
        punto_actual = p_start + desplazamiento

    # Almacenar el resultado en la caché antes de devolverlo
    r_i_j_cache[params_tuple] = punto_actual
    return punto_actual

# Definir la función objetivo a maximizar
def objective(individual):
    m_Njx, m_Njy, r = individual
    return 2 * np.sqrt(m_Njx**2 + m_Njy**2),  # Maximizamos

# Definir las restricciones como penalizaciones
def constraint1(individual):
    m, t, r = 4, 1, individual[2]  # Obtener el valor de r de 'individual'
    s_f = 0.1  # Asignar un valor de ejemplo para s_f
    r_cc = 1 * s_f  # Calcular r_cc
    return max(0, 64 - 2 * (m * (m + 1) * t * np.floor(r / m) + (r % m) * ((r % m) + 1) * t + r_cc))

def constraint2(individual):
    epsilon, d_k = 0.1, 1  # Ejemplo de valores
    m = 2
    expr = (1 / np.sin(np.pi/m)) * (1 / np.sin(np.pi/m)) - epsilon
    return max(0, 36 - (d_k / np.cos((epsilon / 2) * np.sqrt(np.abs(expr)))))

def constraint3(individual):
    r = individual[2]
    return max(0, 3 - r)

def constraint4(individual):
    d_r = 0.1  # Ejemplo de valor
    return max(0, d_r - 0.1)

def constraint5(individual):
    s_f = 0.1  # Ejemplo de valor
    return max(0, s_f)

# Penalización por violación de restricciones
def penalty(individual):
    return constraint1(individual) + constraint2(individual) + constraint3(individual) + constraint4(individual) + constraint5(individual)

# Función de evaluación final
def evaluate(individual):
    obj_value = objective(individual)[0]
    penalties = penalty(individual)
    return np.real(obj_value - penalties),  # Forzamos que sea un número real

# Crear la estructura de los individuos y la población
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizamos
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

# Atributos: m_N,j,x, m_N,j,y son continuos, r es discreto
toolbox.register("attr_m_Njx", random.uniform, 0, 10)
toolbox.register("attr_m_Njy", random.uniform, 0, 10)
toolbox.register("attr_r", random.randint, 0, 3)

# Un individuo es una lista de esos atributos
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_m_Njx, toolbox.attr_m_Njy, toolbox.attr_r), n=1)

# La población es una lista de individuos
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[0, 0, 0], up=[10, 10, 3], eta=0.1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Algoritmo evolutivo con early stopping y multiprocessing
def main():
    # Configuración de paralelización
    pool = Pool()  # Crear un pool de procesos
    toolbox.register("map", pool.map)  # Reemplazar el método map con el de multiprocessing

    population = toolbox.population(n=100)
    NGEN = 40
    CXPB, MUTPB = 0.5, 0.2

    # Variables para early stopping
    improvement_threshold = 1e-5
    last_fitness = None
    generations_without_improvement = 0

    for gen in range(NGEN):
        # Aplicar cruzamiento y mutación
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        
        # Evaluar a la población de manera paralelizada
        fits = toolbox.map(toolbox.evaluate, offspring)

        # Asignar fitness a los individuos
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        # Seleccionar la siguiente generación
        population = toolbox.select(offspring, k=len(population))

        # Early stopping: Verificar si ha habido mejora significativa
        best_ind = tools.selBest(population, 1)[0]
        if last_fitness is not None and abs(best_ind.fitness.values[0] - last_fitness) < improvement_threshold:
            generations_without_improvement += 1
        else:
            generations_without_improvement = 0

        if generations_without_improvement >= 5:  # Parar después de 5 generaciones sin mejora
            print(f"Early stopping at generation: {gen}")
            break

        last_fitness = best_ind.fitness.values[0]

    # Encontrar el mejor individuo
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is:", best_ind)
    print("with fitness:", best_ind.fitness.values[0])

    # Cerrar el pool de procesos
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
