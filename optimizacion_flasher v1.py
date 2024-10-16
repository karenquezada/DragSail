import numpy as np
from deap import base, creator, tools, algorithms
import random
from multiprocessing import Pool
from flasher_lib import *

m = 4
delta = np.pi / 3.7
alpha_variable = np.pi / 2 - np.pi / m
epsilon = 10 * np.pi / 250
eta_variable = (np.pi / m) - (epsilon / 2)
h = 35
distancia_k = dk(h, epsilon, m)
grosor=0.4 #corregir sumando parte de los pliegues -quizas-

def N(r, s_f):
    #calcula el numero de puntos p(i,j)
    r=int(r)
    sum_d = sum(e_distance(i, i + 1) for i in range(r))
    return int((s_f * sum_d) / distancia_k)

r_i_j_cache = {}
def r_i_j(params):
    params_tuple = tuple(params)
    
    if params_tuple in r_i_j_cache:
        return r_i_j_cache[params_tuple]

    i, j, m, i_final, delta, alpha_variable, epsilon, altura, s_f = params  # Añadir s_f a los parámetros
    i_final = int(i_final)
    resultados = []
    distancia_k = dk(altura, epsilon, m)
    punto_inicial = p(0, j, m, delta, alpha_variable, epsilon, s_f)  # Pasar s_f a p
    if np.iscomplexobj(punto_inicial):
        print(f"punto_inicial es complejo: {punto_inicial}")
    punto_actual = punto_inicial 
    modulos_totales = lista_modulos(j, i_final, m, delta, alpha_variable, epsilon, distancia_k, s_f)[0]
    total_modulos = lista_modulos(j, i_final, m, delta, alpha_variable, epsilon, distancia_k, s_f)[2]
    
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
        p_start = p(idx, j, m, delta, alpha_variable, epsilon, s_f)  # Pasar s_f a p
        p_next = p(idx + 1, j, m, delta, alpha_variable, epsilon, s_f)  # Pasar s_f a p
        direccion = (p_next - p_start) / e_distance(p_next, p_start)
        desplazamiento = direccion * remanente
        punto_actual = p_start + desplazamiento
        if np.iscomplexobj(punto_actual):
            print(f"punto_actual es complejo: {punto_actual}")
    r_i_j_cache[params_tuple] = punto_actual
    return punto_actual

# Definir la función objetivo a maximizar
def objective(individual):
    """distancia euclidiana hasta un punto final que define desde 
    un extremo a otro (diametro, pero no área de circulo)"""
    r, sf= individual
    n = N(r, sf)
    m_Njx, m_Njy = r_i_j((n, 0, m, r, delta, alpha_variable, epsilon, h, sf))  
    return 2 * np.sqrt(m_Njx**2 + m_Njy**2)  

def constraint1(individual):
    r, sf = individual  # Obtener el valor de r de 'individual'
    diametroplegado = diametro_plegado(r, 0, m, delta, alpha_variable, epsilon, h, sf, grosor)
    
    # Si el diámetro plegado excede 64, penalizar fuertemente
    if diametroplegado > 60:
        return (diametroplegado - 60) ** 3  # Penalización cuadrática para valores mayores a 64
    else:
        return 0  # No hay penalización si está dentro del límite

# Penalización por violación de restricciones
def penalty(individual):
    return constraint1(individual)

# Función de evaluación final
def evaluate(individual):
    obj_value = objective(individual)
    penalties = penalty(individual)
    if np.iscomplexobj(obj_value):
        print(f"Valor objetivo es complejo: {obj_value}")
    
    if np.iscomplexobj(penalties):
        print(f"Penalizaciones son complejas: {penalties}")
    
    return np.real(obj_value - penalties),

# Función para aplicar las restricciones a los individuos
def apply_constraints(individual):
    # Aquí se pueden aplicar las restricciones específicas, por ejemplo:
    individual[0] = int(round(individual[0]))  # Redondear r
    individual[0] = max(1, min(individual[0], 4))  # Asegurar que esté entre 1 y 3
    sf_lower_bound = 0.0001
    sf_upper_bound = 80

    # Asegúrate de que sf sea positivo y esté dentro de los límites
    individual[1] = max(sf_lower_bound, min(individual[1], sf_upper_bound))


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# Crear la estructura de los individuos y la población
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximizamos
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def generate_positive_sf():
    return random.uniform(0.0001, 80)  # Genera un número en el rango

# Atributos: m_N,j,x, m_N,j,y son continuos, r es discreto
toolbox.register("attr_r", random.randint, 0, 4)
toolbox.register("attr_sf", generate_positive_sf)

# Un individuo es una lista de esos atributos
toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_r, toolbox.attr_sf), n=1)

# La población es una lista de individuos
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

# Operadores genéticos
toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[0, 0.0001, 0.1], up=[3, 4, 5], eta=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)

# Algoritmo evolutivo con early stopping y multiprocessing
def main():
    # Configuración de paralelización
    pool = Pool()  # Crear un pool de procesos
    toolbox.register("map", pool.map)  # Reemplazar el método map con el de multiprocessing

    population = toolbox.population(n=100)
    for i, ind in enumerate(population):
        print(f"Individuo {i}: {ind}")  # Esto imprimirá cada individuo y sus atributos

    NGEN = 40
    CXPB, MUTPB = 0.5, 0.2

    # Variables para early stopping
    improvement_threshold = 1e-5

    last_fitness = None
    generations_without_improvement = 0

    for gen in range(NGEN):
        # Aplicar cruzamiento y mutación
        print(f"Generation: {gen}")
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        print("Offspring generated", offspring)

        # Aplicar restricciones a los individuos generados
        for ind in offspring:
            apply_constraints(ind)

        # Evaluar a la población de manera paralelizada
        fits = toolbox.map(toolbox.evaluate, offspring)

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
    optimal_objective_value = objective(best_ind)  # Llama a la función objetivo directamente
    print("Optimal objective value:", optimal_objective_value)
    constraint_value=constraint1(best_ind)
    print("Constraint value:", constraint_value)
    diametro=diametro_plegado(best_ind[0], 0, m, delta, alpha_variable, epsilon, h, best_ind[1], grosor)
    print("Diametro plegado:", diametro)
    

    # Cerrar el pool de procesos
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
