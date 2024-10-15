import numpy as np
from deap import base, creator, tools, algorithms
import random
from multiprocessing import Pool
from miura_drawer import miura_drawer

def custom_crossover(ind1, ind2):
    # Cruza las variables continuas (w y h)
    for i in range(2):  # Los dos primeros elementos son continuos
        if random.random() < 0.5:  # Crossover simple
            ind1[i], ind2[i] = ind2[i], ind1[i]

    # Asegúrate de que alpha, nv, y nh sigan siendo múltiplos de 0.5
    ind1[2] = int(round(ind1[2]))  # alpha (entero)
    ind2[2] = int(round(ind2[2]))  # alpha (entero)
    
    ind1[3] = round(ind1[3] * 2) / 2  # nv (múltiplo de 0.5)
    ind2[3] = round(ind2[3] * 2) / 2  # nv (múltiplo de 0.5)
    ind1[4] = round(ind1[4] * 2) / 2  # nh (múltiplo de 0.5)
    ind2[4] = round(ind2[4] * 2) / 2  # nh (múltiplo de 0.5)

    return ind1, ind2
# Definir la función objetivo
def objetivo(theta):
    w, h, alpha, nv, nh = theta[0], theta[1], theta[2], theta[3], theta[4]
    d = h / np.tan(np.radians(alpha))  # Convertir alpha a radianes
    
    # Calcular el área plegada
    area_plegada = (w * h / (h**2 + d**2)) * (d**2 + (2 * w * nh - 3 * w) * d + h**2)
    # if area_plegada < 0:
    #     area_plegada = 0

    # Cálculo del resultado base
    resultado_final = np.sqrt(max(0, 20 * (w * h * nv * nh))) + (5 * np.sqrt(area_plegada))
    
    # Inicializar penalización
    penalizacion = 0

    restr1 = restriccion1(theta)
    restr2 = restriccion2(theta)
    restr3 = restriccion3(theta)
    restr4 = restriccion4(theta)
    restr5 = restriccion5(theta)
    restr6 = restriccion6(theta)

    print(f"Evaluando theta: {theta}")
    print(f"Restricciones: 1: {restr1}, 2: {restr2}, 3: {restr3}, 4: {restr4}, 5: {restr5}, 6: {restr6}")

    if restr1 < 0:  # ancho
        penalizacion += abs(restr1) * 10000
    
    if restr2 < 0:  # largo
        penalizacion += abs(restr2) * 10000
    
    if restr3 < 0:  # superposición
        penalizacion += abs(restr3) * 1000
    
    if restr4 < 0:  # alto
        penalizacion += abs(restr4) * 10000
    
    if restr5 < 0:  # separaciones
        penalizacion += abs(restr5) * 500  # Penalización menor
    
    if restr6 < 0:  # alto desplegado
        penalizacion += abs(restr6) * 500  # Penalización menor

    # Sumar la penalización al resultado final
    resultado_final += penalizacion

    return resultado_final,


def restriccion1(theta):
    #ancho plegado cabe en el cubesat
    #considerando 15 mm de espacio disponible
    nv=theta[3]
    t=0.5
    return 15-(4*nv*t)

def restriccion2(theta): 
    #largo plegado cabe en el cubesat
    nh, w, alpha = theta[4], theta[0], theta[2]
    return 200 - (nh * np.sqrt(2 * (1 - np.cos(np.pi - 2 * np.radians(alpha)))) * w + w + w * np.cos(np.radians(alpha)))

def restriccion3(theta): 
    #no hay superposicion de capas
    w, h, alpha = theta[0], theta[1], theta[2]
    return w * np.sqrt(2 * (1 + np.cos(2 * np.radians(alpha)))) - (h / np.sin(np.radians(alpha)))

def restriccion4(theta):  
    #alto plegado cabe en el cubesat
    w, alpha = theta[0], theta[2]
    return 100 - w * np.sin(np.radians(alpha))

def restriccion5(theta): 
    #no hay separaciones al plegar
    w, h = theta[0], theta[1]
    return h - w

def restriccion6(theta):
    #alto desplegado es menor a 300 (depende del mecanismo de despliegue) 
    w, h, nv, nh, alpha = theta[0], theta[1], theta[3], theta[4], theta[2]
    return 300 - 2 * h * nv

# Función para aplicar restricciones
def apply_constraints(individual):
    # Restringir w (ancho) y h (altura) a [1, 100]
    individual[0] = max(10, min(individual[0], 15))
    individual[1] = max(10, min(individual[1], 25))
    # Restringir alpha a [1, 90]
    individual[2] = (max(40, min(individual[2], 60)))
    # Restringir nv (número de divisiones verticales) y nh (número de divisiones horizontales) a [1, 10]
    individual[3] = max(2, min(round(individual[3] * 2) / 2, 6))  # nv
    individual[4] = max(2, min(round(individual[4] * 2) / 2, 6))  # nh

SEED = 42
# random.seed(SEED)
# np.random.seed(SEED)

# Crear el tipo de problema de optimización
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def generate_positive_w():
    return random.uniform(10, 15)  # Ajusta el rango según tus necesidades

def generate_positive_h():
    return random.uniform(10, 25)  # Ajusta el rango según tus necesidades

def generate_positive_alpha():
    return random.uniform(40, 60)  # Ajusta el rango según tus necesidades

def generate_integer_nv():
    return random.randint(2, 10)  # Ajusta el rango según tus necesidades

def generate_integer_nh():
    return random.randint(2, 10)  # Ajusta el rango según tus necesidades

toolbox.register("attr_w", generate_positive_w)
toolbox.register("attr_h", generate_positive_h)
toolbox.register("attr_alpha", generate_positive_alpha)
toolbox.register("attr_nv", generate_integer_nv)
toolbox.register("attr_nh", generate_integer_nh)

toolbox.register("individual", tools.initCycle, creator.Individual,
                 (toolbox.attr_w, toolbox.attr_h, toolbox.attr_alpha, toolbox.attr_nv, toolbox.attr_nh), n=1)

toolbox.register("population", tools.initRepeat, list, toolbox.individual)

#toolbox.register("mate", tools.cxBlend, alpha=0.5)
toolbox.register("mate", custom_crossover)
toolbox.register("mutate", tools.mutPolynomialBounded, low=[0, 0.0001, 0.1], up=[3, 4, 5], eta=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", objetivo)


# Función principal
def main():
    # Crear un pool de procesos para paralelización
    pool = Pool()
    #toolbox.register("map", pool.map)

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.2)

    # Población inicial
    population = toolbox.population(n=100)
    
    NGEN = 100
    CXPB, MUTPB = 0.5, 0.2
    
    # Variables para early stopping
    improvement_threshold = 1e-5
    last_fitness = None
    generations_without_improvement = 0

    for gen in range(NGEN):
        print(f"Generation: {gen}")

        # Aplicar cruzamiento y mutación
        offspring = algorithms.varAnd(population, toolbox, cxpb=CXPB, mutpb=MUTPB)

        # Aplicar restricciones a los individuos generados
        for ind in offspring:
            apply_constraints(ind)  # Asegúrate de que los individuos respeten los límites

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
    miura_drawer(best_ind[0], best_ind[1], best_ind[2], best_ind[3], best_ind[4])
    # Cerrar el pool de procesos
    pool.close()
    pool.join()



if __name__ == "__main__":
    main()
