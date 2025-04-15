import numpy as np
import os
import glob

ruta_archivos = "c:/Users/Karen/Desktop/repo_memoria/Solar-Sail/data_miura"
patron = os.path.join(ruta_archivos, "fitness_MUTPB_*_Miura_*.txt")

mutpb_vals = sorted({float(os.path.basename(f).split('_')[2]) for f in glob.glob(patron)})
cxpb_vals = sorted({float(os.path.basename(f).split('_')[4].replace('.txt','')) for f in glob.glob(patron)})

# Crear matrices vacías para los resultados
mejor_fitness_matrix = np.full((len(mutpb_vals), len(cxpb_vals)), np.nan)
promedio_fitness_matrix = np.full((len(mutpb_vals), len(cxpb_vals)), np.nan)


for archivo in glob.glob(patron):
    nombre = os.path.basename(archivo)
    partes = nombre.replace('.txt', '').split('_')
    mutpb = float(partes[2])
    cxpb = float(partes[4])


    try:
        data = np.genfromtxt(archivo, skip_header=1)
        if data.ndim == 1:
            data = data[np.newaxis, :]
        ultima_fila = data[-1]
        mejor = ultima_fila[-2]
        promedio = ultima_fila[-1]

        i = mutpb_vals.index(mutpb)  # Fila
        j = cxpb_vals.index(cxpb)    # Columna

        mejor_fitness_matrix[i, j] = mejor
        promedio_fitness_matrix[i, j] = promedio

    except Exception as e:
        print(f"Error leyendo {archivo}: {e}")


# np.savetxt("mejor_fitness_matrix.csv", mejor_fitness_matrix, delimiter=",")
# np.savetxt("promedio_fitness_matrix.csv", promedio_fitness_matrix, delimiter=",")

# También puedes imprimirlas o graficarlas más adelante
print("Matriz de mejor fitness:")
print(mejor_fitness_matrix)

print("\nMatriz de fitness promedio:")
print(promedio_fitness_matrix)
