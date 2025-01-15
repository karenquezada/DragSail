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

    # Cálculo del resultado base
    resultado_final = np.sqrt(max(0, 5*(w**0.6 * h**0.6 * nv**0.4 * nh**0.4))) - (5 * np.sqrt(area_plegada))
    return resultado_final

def restriccion1(theta):
    #ancho plegado cabe en el cubesat
    #considerando 15 mm de espacio disponible
    nv=theta[3]
    t=0.1
    dif=13.2-(4*nv*t)
    if dif<0:
        return abs(dif)**7
    else:
        return 0

def restriccion2(theta): 
    # largo plegado cabe en el cubesat
    nh, w, alpha = theta[4], theta[0], theta[2]
    dif = 200 - (nh * np.sqrt(2 * (1 - np.cos(np.pi - 2 * np.radians(alpha)))) * w + w + w * np.cos(np.radians(alpha)))
    if dif < 0:
        return abs(dif) ** 10
    else:
        return 0

def restriccion3(theta): 
    # no hay superposicion de capas
    w, h, alpha = theta[0], theta[1], theta[2]
    dif = w * np.sqrt(2 * (1 + np.cos(2 * np.radians(alpha)))) - (h / np.sin(np.radians(alpha)))
    if dif < 0:
        return abs(dif) ** 5
    else:
        return 0

def restriccion4(theta):  
    #alto cabe en el cubesat esto es por trigonometria
    w, alpha = theta[0], theta[2]
    dif = 100 - w * np.sin(np.radians(alpha))
    if dif < 0:
        return abs(dif) ** 5
    else:
        return 0

def restriccion5(theta): 
    # no hay separaciones al plegar
    w, h = theta[0], theta[1]
    dif = h - w
    if dif < 0:
        return abs(dif) ** 2  # Penalización menor
    else:
        return 0

# def restriccion6(theta):
#     # #alto desplegado es menor a 300 (depende del mecanismo de despliegue) 
#     # w, h, nv, nh, alpha = theta[0], theta[1], theta[3], theta[4], theta[2]
#     # dif = 300 - 2 * h * nv
#     # if dif < 0:
#     #     return abs(dif) ** 3  # Penalización menor
#     # else:
#     #     return 0

def calcular_difs(theta):
    nv, nh, w, h, alpha = theta[3], theta[4], theta[0], theta[1], theta[2]
    #ancho, largo, no superposicion, alto, no separaciones, alto desplegado
    dif1 = 15 - (4 * nv * 0.1)
    dif2 = 200 - (nh * np.sqrt(2 * (1 - np.cos(np.pi - 2 * np.radians(alpha)))) * w + w + w * np.cos(np.radians(alpha)))
    dif3 = w * np.sqrt(2 * (1 + np.cos(2 * np.radians(alpha)))) - (h / np.sin(np.radians(alpha)))
    dif4 = 100 - w * np.sin(np.radians(alpha))
    dif5 = h - w
    #dif6 = 300 - 2 * h * nv
    
    return dif1, dif2, dif3, dif4, dif5

def penalty(theta):
    return restriccion1(theta) + restriccion2(theta) + restriccion3(theta) + restriccion4(theta) + restriccion5(theta)
 
def evaluate(theta):
    obj_value = objetivo(theta)
    penalties = penalty(theta)
    return obj_value - penalties,

# Función para aplicar restricciones
def apply_constraints(individual):
    #w, h, alpha, nv, nh = theta[0], theta[1], theta[2], theta[3], theta[4]
    # Restringir w (ancho) y h (altura) a [1, 100]
    individual[0] = max(20, min(individual[0], 40))
    individual[1] = max(20, min(individual[1], 40))
    # Restringir alpha a [1, 90]
    individual[2] = (max(40, min(individual[2], 60)))
    # Restringir nv (número de divisiones verticales) y nh (número de divisiones horizontales) a [1, 10]
    individual[3] = max(4, min(round(individual[3] * 2) / 2, 30))  # nv
    individual[4] = max(4, min(round(individual[4] * 2) / 2, 30))  # nh

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

# Crear el tipo de problema de optimización
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)

toolbox = base.Toolbox()

def generate_positive_w():
    return random.uniform(30, 40)  # Ajusta el rango según tus necesidades

def generate_positive_h():
    return random.uniform(5, 25)  # Ajusta el rango según tus necesidades

def generate_positive_alpha():
    return random.uniform(40, 60)  # Ajusta el rango según tus necesidades

def generate_integer_nv():
    return random.randint(4, 30)  # Ajusta el rango según tus necesidades

def generate_integer_nh():
    return random.randint(4, 30)  # Ajusta el rango según tus necesidades

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
#toolbox.register("mutate", tools.mutPolynomialBounded, low=[0, 0.0001, 0.1], up=[3, 4, 5], eta=1, indpb=0.1)
toolbox.register("select", tools.selTournament, tournsize=3)
toolbox.register("evaluate", evaluate)


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
        #print(f"Generation: {gen}")

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

        if generations_without_improvement >= 15:  # Parar después de 5 generaciones sin mejora
            print(f"Early stopping at generation: {gen}")
            break

        last_fitness = best_ind.fitness.values[0]

    # Encontrar el mejor individuo
    best_ind = tools.selBest(population, 1)[0]
    print("Best individual is:", best_ind)
    print("with fitness:", best_ind.fitness.values[0])
    restricciones=calcular_difs(best_ind)
    print("Restricciones",restricciones)
    miura_drawer(best_ind[0], best_ind[1], best_ind[2], best_ind[3], best_ind[4])
    # Cerrar el pool de procesos
    pool.close()
    pool.join()



if __name__ == "__main__":
    main()
                                                                                                                                                               