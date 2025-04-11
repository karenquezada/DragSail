import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def area(a, b, mg, ms, angulo1):
    angulo1 = np.radians(angulo1)
    g = 4 * mg * b * np.sin(angulo1)
    s = 4 * ms * a + b * np.sin(np.pi/2 - angulo1)
    return g * s

def largo(a, ms, angulo1):
    angulo1 = np.radians(angulo1)
    beta = np.pi - 2 * angulo1
    c = np.sqrt(2 * (a**2) + 2 * (a**2) * np.cos(beta))
    ns = 2 * ms
    return c * ns

def aristas(a, b, ms, mg, angulo1):
    angulo1 = np.radians(angulo1)
    g = 4 * mg * b * np.sin(angulo1)
    s = 4 * ms * a + b * np.sin(np.pi/2 - angulo1)
    return g,s

def objective(vars):
    a, b, ms, mg, alpha = vars
    return -area(a, b, ms, mg, alpha)

#theta = 85
bounds = opt.Bounds([20, 20, 5, 1, 30], [100, 100, 100, 100, 85])

def square_panel(vars):
    a, b, ms, mg, alpha = vars
    return abs(ms - mg)

grosor = 1  # mm

def in_cubesat(vars):
    a, b, ms, mg, alpha = vars
    ancho = 8 * mg * grosor
    largo_value = largo(a, ms, alpha)
    return min(100 - ancho, 100 - largo_value)

def b_less_than_3a(vars):
    a, b, ms, mg, alpha = vars
    return 2 * a - b


non_linear_constraints1 = {'type': 'ineq', 'fun': square_panel}
non_linear_constraints2 = {'type': 'ineq', 'fun': in_cubesat}
non_linear_constraints3 = {'type': 'ineq', 'fun': b_less_than_3a}

# Lista de posibles soluciones iniciales
x0 = [25,25,25,25,30]

result = opt.minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=[non_linear_constraints1, non_linear_constraints2, non_linear_constraints3], options={'maxiter': 5000})

if result.success:
    a_opt, b_opt, ms_opt, mg_opt, alpha_opt = result.x
    total_area = area(a_opt, b_opt, ms_opt, mg_opt, alpha_opt)
    print("Solución óptima encontrada:")
    print("a =", a_opt, "b =", b_opt, "ms =", ms_opt, "mg =", mg_opt, "alpha =", alpha_opt)
    print("Valor de la función objetivo:", -result.fun)
    print("Largo de los lados:", aristas(a_opt, b_opt, ms_opt, mg_opt, alpha_opt))
else:
    print("No se encontró una solución óptima.")
    print("Mensaje del resultado", result.message)

print("Evaluación de las restricciones:")
print("square_panel:", square_panel(result.x))
print("in_cubesat:", in_cubesat(result.x))
print("b_less_than_3a:", b_less_than_3a(result.x))

print("\n")

# Definir valores fijos para las otras variables
ms_fixed = ms_opt
mg_fixed = mg_opt
alpha_fixed = alpha_opt

# Rango de valores para 'a' y 'b'
a_vals = np.linspace(bounds.lb[0], bounds.ub[0], 100)
b_vals = np.linspace(bounds.lb[1], bounds.ub[1], 100)
ms_vals = np.linspace(bounds.lb[2], bounds.ub[2], 100)
mg_vals = np.linspace(bounds.lb[3], bounds.ub[3], 100)
alpha_vals = np.linspace(bounds.lb[4], bounds.ub[4], 100)

# Calcular el área para cada combinación de 'a' y 'b'
area_vals = np.zeros((len(a_vals), len(b_vals)))
for i, a in enumerate(a_vals):
    for j, b in enumerate(b_vals):
        area_vals[i, j] = area(a, b, ms_fixed, mg_fixed, alpha_fixed)

# Crear la primera figura y gráfico de contorno para el área en función de a y b
plt.figure()
plt.contourf(a_vals, b_vals, area_vals, cmap='viridis')
plt.colorbar(label='Área')
plt.xlabel('a')
plt.ylabel('b')
plt.title('Área en función de a y b')

# Calcular el área para cada combinación de 'ms' y 'mg'
area_vals2 = np.zeros((len(ms_vals), len(mg_vals)))
for i, ms in enumerate(ms_vals):
    for j, mg in enumerate(mg_vals):
        area_vals2[i, j] = area(a_opt, b_opt, ms, mg, alpha_fixed)

# Crear la segunda figura y gráfico de contorno para el área en función de ms y mg
plt.figure()
plt.contourf(ms_vals, mg_vals, area_vals2, cmap='viridis')
plt.colorbar(label='Área')
plt.xlabel('ms')
plt.ylabel('mg')
plt.title('Área en función de ms y mg')

# Calcular el área para cada combinación de 'alpha' y 'ms'
area_vals3 = np.zeros((len(alpha_vals), len(ms_vals)))
for i, alpha in enumerate(alpha_vals):
    for j, ms in enumerate(ms_vals):
        area_vals3[i, j] = area(a_opt, b_opt, ms, mg_fixed, alpha)

# Crear la tercera figura y gráfico de contorno para el área en función de alpha y ms
plt.figure()
plt.contourf(alpha_vals, ms_vals, area_vals3, cmap='viridis')
plt.colorbar(label='Área')
plt.xlabel('alpha')
plt.ylabel('ms')
plt.title('Área en función de alpha y ms')

plt.show()
