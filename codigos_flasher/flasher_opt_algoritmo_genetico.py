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
def main():
    # Configuración de paralelización
    pool = Pool()  # Crear un pool de procesos
    toolbox.register("map", pool.map)  # Reemplazar el método map con el de multiprocessing
    n_individuos=500

    population = toolbox.population(n_individuos)
   

    NGEN = 1000
    CXPB, MUTPB = 0.2, 0.3

    # Variables para early stopping
    improvement_threshold = 1e-5

    #Listas para graficar la evolucion de las soluciones.
    last_fitness = None
    avg_fitness_per_gen=[]
    best_fitness_per_gen=[]
    std_fitness_per_gen=[]

    for gen in range(NGEN):
        #elitismo en la gen actual
        elite = tools.selBest(population, 1)[0]
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)
        #print("Offspring generated", offspring)

        # Aplicar restricciones a los individuos generados
        for ind in offspring:
            apply_constraints(ind)

        # Evaluar a la población de manera paralelizada
        fits = toolbox.map(toolbox.evaluate, offspring)

        for fit, ind in zip(fits, offspring):
            ind.fitness.values = fit

        #reemplazo del peor individuo
        worst_index = np.argmin([ind.fitness.values[0] for ind in offspring])
        offspring[worst_index] = toolbox.clone(elite)

        # Seleccionar la siguiente generación
        population = toolbox.select(offspring, k=len(population))

        # Early stopping: Verificar si ha habido mejora significativa
        best_ind = tools.selBest(population, 1)[0]
        best_fitness_per_gen.append(best_ind.fitness.values[0])
        avg_fitness_per_gen.append(np.mean([ind.fitness.values[0] for ind in population]))
        std_fitness_per_gen.append(np.std([ind.fitness.values[0] for ind in population]))



        print(f"GENERACION {gen}, Mejor fitness: {best_fitness_per_gen[-1]:.4f} Promedio de fitness :{avg_fitness_per_gen[-1]:.4f} Desv estándar {std_fitness_per_gen[-1]:.4f}")

        #print(f"Generación {gen}: mejor fitness = {max(fits_values):.4f}, fitness promedio = {avg_fitness:.4f} ") 
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
    
    plt.figure(figsize=(8,5))
    generations = range(1, len(best_fitness_per_gen))
    generaciones_a_eliminar = 2
    # plt.plot(np.abs(best_fitness_per_gen[1:]), label="Mejor Fitness", color="blue")
    # plt.plot(np.abs(avg_fitness_per_gen[1:]), label="Fitness Promedio", color="orange")
    plt.plot((best_fitness_per_gen[generaciones_a_eliminar:]), label="Mejor Fitness", color="blue")
    plt.plot((avg_fitness_per_gen[generaciones_a_eliminar:]), label="Fitness Promedio", color="orange")
    plt.plot(std_fitness_per_gen[generaciones_a_eliminar:], label="Desviación Estándar", color="green")

    plt.xlabel("Generaciones")
    plt.ylabel("Valor de Fitness")
    plt.title("Evolución de la Solución, origami Flasher")
    param_text = (
    f"N° Individuos: {n_individuos}\n"
    f"N° Generaciones: {NGEN}\n"
    f"Tasa de Cruza (CXPB): {CXPB}\n"
    f"Tasa de Mutación (MUTPB): {MUTPB}"
)

# Añadir como caja de texto en la parte superior derecha
    plt.text(0.98, 0.81, param_text,
         transform=plt.gca().transAxes,
         fontsize=10, va='bottom', ha='right',
         bbox=dict(boxstyle="round,pad=0.4", facecolor="lightgray", edgecolor="black", alpha=0.8))

    plt.legend()
    plt.grid()
    plt.yscale('log')  # Cambiar el eje Y a logarítmico
    plt.xlim(generaciones_a_eliminar, max(generations))
    plt.show()

    # Cerrar el pool de procesos
    pool.close()
    pool.join()

if __name__ == "__main__":
    main()
