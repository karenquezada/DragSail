import numpy as np
import matplotlib.pyplot as plt

#para cxpb=0,1
mjr_fitness_01=[411.15733893, 411.15733892, 411.15733882, 411.15733889, 411.15733893, 411.12389057, 411.15733620, 402.72004409, 411.15576947]
fitness_prom_01=[411.15733161, 411.15733022, 411.15727702, 411.15599999, 410.16795795, 410.44893051, 411.15729480, 401.70721769, 404.58664188]

#para cxpb=0,2
mjr_fitness_02=[411.15733893, 411.15733893, 411.15733893, 411.15733893, 411.15733893, 411.15733893, 401.86786506, 409.11239943, 411.15733882]
fitness_prom_02=[411.15733161, 411.15733829, 411.15733795, 411.15733042, 411.15733215, 410.78700056, 363.58126645, 382.71008998, 410.78682049]

#para cxpb=0,3
mjr_fitness_03=[411.15733893, 411.15733878, 411.15733893, 411.08137965, 411.15733893, 411.15733853, 411.15733892, 411.15733879, 411.15733893]
fitness_prom_03=[411.15733873, 411.15615532, 411.15733673, 379.49619602, 409.79763814, 410.74668546, 410.78695151, 411.15726275, 411.15720482]

#para cxpb=0,4
mjr_fitness_04=[411.15733893, 411.15733893, 411.15733893, 411.15733885, 411.15733893, 411.15733892, 405.62921534, 411.15733893, 411.15733892]
fitness_prom_04=[411.15733716, 411.15733698, 411.15733065, 411.15582076, 411.15730805, 411.15733033, 387.01744558, 409.42729822, 408.68662641]

#para cxpb=0,5
mjr_fitness_05=[410.90823492, 411.15733893, 411.15733893, 411.15733893, 411.15733893, 411.15733893, 411.15733893, 411.15733893, 410.67865087]
fitness_prom_05=[392.89315645, 411.15732830, 411.15725112, 411.15730543, 411.15732395, 410.78691976, 409.42722998, 410.51857473, 347.90411015]

#para cxpb=0,6
mjr_fitness_06=[411.15733893, 411.15733891, 411.15733893, 411.15733893, 411.15733893, 411.15733893, 411.15733893, 411.15733893, 411.15733893]
fitness_prom_06=[411.15730794, 411.15727285, 411.15733170, 411.15728641, 410.41665218, 410.41657032, 410.78700284, 410.27380653, 410.30395646]

#para cxpb=0,7
mjr_fitness_07=[411.15733893, 410.93935589, 411.15733133, 411.09429637, 409.50360429, 411.09996022, 410.99438052, 411.15668500, 411.15733893]
fitness_prom_07=[411.15731217, 362.37776157, 403.02367973, 384.39748152, 384.61083428, 400.81974916, 380.79561446, 375.18515150, 409.74040540]

#para cxpb=0,8
mjr_fitness_08=[411.15713464, 410.57260802, 411.15726851, 411.15670416, 401.43743558, 411.15502617, 410.96416580, 410.45502221, 386.14724614]
fitness_prom_08=[405.25474812, 384.47053261, 393.92001349, 401.43743558, 380.79748843, 366.91488406, 330.25349608, 334.71343297, 355.97118048]

#para cxpb=0,9
mjr_fitness_09=[401.81299834, 411.15721153, 411.15733893, 411.14776557, 411.15255651, 411.13083571, 410.89157196, 411.15321905, 411.15733397]
fitness_prom_09=[351.29886115, 340.10181440, 411.15728724, 342.94706362, 359.64398615, 345.25687250, 332.66901798, 344.41027451, 409.91189898]


cxpbs = np.linspace(0.1, 0.9, 9)  
mutpbs = np.linspace(0.1, 0.9, 9)

# Crear una matriz con los mejores fitness
best_fitness = np.array([
    mjr_fitness_01,
    mjr_fitness_02,
    mjr_fitness_03,
    mjr_fitness_04,
    mjr_fitness_05,
    mjr_fitness_06,
    mjr_fitness_07,
    mjr_fitness_08,
    mjr_fitness_09
])

# Crear una matriz con los fitness promedio
average_fitness = np.array([
    fitness_prom_01,
    fitness_prom_02,
    fitness_prom_03,
    fitness_prom_04,
    fitness_prom_05,
    fitness_prom_06,
    fitness_prom_07,
    fitness_prom_08,
    fitness_prom_09
])

# Crear el gráfico de contorno para el fitness promedio
plt.figure(figsize=(8, 6))
# Crear la malla de valores X, Y
X, Y = np.meshgrid(cxpbs, mutpbs)


contour_avg = plt.contourf(X, Y, average_fitness, cmap='gist_rainbow_r', levels=100)
plt.colorbar(contour_avg, label='Fitness Promedio')
plt.title('Fitness Promedio origami Flasher')
plt.xlabel('Parámetro de cruza (cxpb)')
plt.ylabel('Parámetro de mutación (mutpb)')
plt.show()

#gist_rainbow_r
# Crear el gráfico de contorno
plt.figure(figsize=(8, 6))
#mosaico = plt.pcolormesh(X, Y, best_fitness, shading='auto', cmap='plasma')
contour = plt.contourf(X, Y, best_fitness, cmap='gist_rainbow_r', levels=100)
plt.colorbar(contour, label='Mejor Fitness')
#plt.colorbar(mosaico, label='Mejor Fitness')
plt.title('Mejores Fitness origami Flasher')
plt.xlabel('Parámetro de cruza (cxpb)')
plt.ylabel('Parámetro de mutación (mutpb)')
plt.show()