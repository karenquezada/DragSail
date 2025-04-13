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

# Algoritmo evolutivo con early stopping y multiprocessing
def main(MUTPB, mutpb_label, CXPB):
    pool = Pool()
    toolbox.register("map", pool.map)

    n_individuos = 500
    population = toolbox.population(n_individuos)

    NGEN = 1000
    improvement_threshold = 1e-5

    last_fitness = None
    generations_without_improvement = 0

    best_fitness_per_gen = []
    avg_fitness_per_gen = []

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

        fitness_values = [ind.fitness.values[0] for ind in population]
        best_fitness = max(fitness_values)
        avg_fitness = sum(fitness_values) / len(fitness_values)

        best_fitness_per_gen.append(best_fitness)
        avg_fitness_per_gen.append(avg_fitness)

        if last_fitness is not None and abs(best_fitness - last_fitness) < improvement_threshold:
            generations_without_improvement += 1
        else:
            generations_without_improvement = 0

        if generations_without_improvement >= 5:
            break

        last_fitness = best_fitness

    pool.close()
    pool.join()

    # Guardar resultados en archivo .txt
    with open(f"fitness_MUTPB_{mutpb_label}_Flasher_{CXPB}.txt", "w") as f:
        f.write("Generación\tMejor_Fitness\tFitness_Promedio\n")
        for i, (b, a) in enumerate(zip(best_fitness_per_gen, avg_fitness_per_gen)):
            f.write(f"{i}\t{b:.8f}\t{a:.8f}\n")

    return best_fitness_per_gen, avg_fitness_per_gen

if __name__ == "__main__":
    mutpb_values = np.arange(0.1, 1.0, 0.1)
    cxpb_values= np.arange(0.5, 1.0, 0.1)
    print(cxpb_values)
    print(mutpb_values)
    best_fitness_by_mutpb = []
    for cruza in cxpb_values:
        for mutpb in mutpb_values:
            print(f"Ejecutando para MUTPB = {mutpb:.1f}")
            best_fitness_per_gen, avg_fitness_per_gen = main(mutpb, f"{mutpb:.1f}",cruza)
            best_fitness_by_mutpb.append(best_fitness_per_gen[-1])  # último valor de mejor fitness


    # Resumen general
        print("\nResumen de mejores fitness por valor de MUTPB:")
        for mutpb, fit in zip(mutpb_values, best_fitness_by_mutpb):
            print(f"MUTPB={mutpb:.1f}: fitness={fit:.8f}")

    # Graficar fitness final por MUTPB
    # plt.figure()
    # plt.plot(mutpb_values, best_fitness_by_mutpb, marker='o')
    # plt.xlabel("MUTPB")
    # plt.ylabel("Mejor Fitness Final")
    # plt.title("Efecto de MUTPB sobre el Mejor Fitness")
    # plt.grid(True)
    # plt.tight_layout()
    # plt.savefig("comparacion_mutpb_fitness_final.png")
    # plt.show()