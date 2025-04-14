import numpy as np
from deap import base, creator, tools, algorithms
import random
from multiprocessing import Pool
from flasher_lib import *
import matplotlib.pyplot as plt

m = 4
delta = np.pi / 3.7
alpha_variable = np.pi / 2 - np.pi / m
epsilon = 0.05236
eta_variable = (np.pi / m) - (epsilon / 2)
h = 20
distancia_k = dk(h, epsilon, m)
grosor=0.1

# Definir la función objetivo a maximizar
def objective(individual):
    """distancia euclidiana hasta un punto final que define desde 
    un extremo a otro (diametro, pero no área de circulo)"""
    r, sf= individual
    m_Njx, m_Njy = p(r, 0, m, delta, alpha_variable, epsilon, sf) 
    return 2 * np.sqrt(m_Njx**2 + m_Njy**2)  

def constraint1(individual):
    r, sf = individual  # Obtener el valor de r de 'individual'
    diametroplegado = diametro_plegado(r, 0, m, delta, alpha_variable, epsilon, h, sf, grosor)

    if diametroplegado > 61:
        return (diametroplegado - 61) ** 3 
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
    individual[0] = int(round(individual[0]))  # Redondear r
    individual[0] = max(1, min(individual[0], 7))  # Asegurar que esté entre 1 y 3
    sf_lower_bound = 0.0001
    sf_upper_bound = 80

   
    individual[1] = max(sf_lower_bound, min(individual[1], sf_upper_bound))


SEED = 42
random.seed(SEED)
np.random.seed(SEED)


# Crear la estructura de los individuos y la población
creator.create("FitnessMax", base.Fitness, weights=(1.0,))  
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def generate_positive_sf():
    return random.uniform(0.0001, 80)  # Genera un número en el rango

# Atributos: m_N,j,x, m_N,j,y son continuos, r es discreto
toolbox.register("attr_r", random.randint, 1, 7)
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

NGEN = 300
CXPB, MUTPB = 0.2, 0.3
population_sizes = [30, 50, 100, 500, 1000, 1500, 2000]

resultados_mejor = {}
resultados_promedio = {}

def main(population_si):
    pool=Pool()
    toolbox.register("map", pool.map)
    n_individuos = population_si
    population = toolbox.population(n_individuos)
    NGEN = 1000
    improvement_threshold = 1e-5
    last_fitness = None
    generations_without_improvement = 0

    for gen in range(NGEN):
        elite = tools.selBest(population, 1)[0]
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

        for ind in offspring:
            apply_constraints(ind)
    
        fits = toolbox.map(toolbox.evaluate, offspring)
        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        worst_index = np.argmin([ind.fitness.values[0] for ind in offspring])
        offspring[worst_index] = toolbox.clone(elite)
        population = toolbox.select(offspring, k=len(population))

        best_ind = tools.selBest(population, 1)[0]
        best_fitness = best_ind.fitness.values[0]
        
        if last_fitness is not None and abs(best_fitness - last_fitness) < improvement_threshold:
            generations_without_improvement += 1
        else:
            generations_without_improvement = 0
        
        if generations_without_improvement >= 15:
            break

        last_fitness = best_fitness

    best_ind = tools.selBest(population, 1)[0]
    avg_fitness = np.mean([ind.fitness.values[0] for ind in population])

    pool.close()
    pool.join()

    return best_ind.fitness.values[0], avg_fitness
       
if __name__ == "__main__":
    for population_size in population_sizes:
        best_fitness, avg_fitness = main(population_size)
        resultados_mejor[population_size] = abs(best_fitness)

        print(f"Mejor fitness para población {population_size}: {best_fitness:.8f}")
        resultados_promedio[population_size] = abs(avg_fitness)

    plt.figure(figsize=(10, 6))
    plt.plot(population_sizes, list(resultados_mejor.values()), marker='o', label='Mejor Fitness')
    plt.yscale('log')  # Escala logarítmica para el eje y
    plt.title("Mejor Fitness por Generación, origami Flasher")
    plt.xlabel("Generación")
    plt.ylabel("Mejor Fitness (Escala Logarítmica)")
    plt.legend()
    plt.grid()
    plt.show()