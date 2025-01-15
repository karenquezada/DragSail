import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt


#mh=horizontal, mv=vertical
def area(a, b, mv, mh, angulo1):
    angulo1 = np.radians(angulo1)
    largo= 2*a+b*np.cos(angulo1)
    ancho= 2*b*np.sin(angulo1)
    triangulo1=(2*b**2*np.sin(angulo1)*np.cos(angulo1))/2
    triangulo2=(b**2*np.sin(angulo1)*np.cos(angulo1))
    return (largo*ancho-triangulo1-triangulo2)*mh*mv

def largo_plegado(a, b, mv, angulo1):
    angulo1 = np.radians(angulo1)
    beta = np.pi - 2 * angulo1
    c = np.sqrt(2 * (a**2) + 2 * (a**2) * np.cos(beta))
    return mv*c+b


def square_panel(vars):
    a, b, mv, mh, alpha = vars
    return abs(mv - mh)


def in_cubesat_ancho(vars):
    a, b, mv, mh, alpha = vars
    ancho = 8 * mh
    return 100 - ancho

def in_cubesat_largo(vars):
    a, b, mv, mh, alpha = vars
    largo_value = largo_plegado(a, b, mv, alpha)
    return 100-largo_value

def aristas(a, b, mv, mh, angulo1):
    angulo1 = np.radians(angulo1)
    g = 4 * mh * b * np.sin(angulo1)
    s = 4 * mv * a + b * np.sin(np.pi/2 - angulo1)
    return g,s

def ratio(a,b,mv,alpha):
    largo_desplegado=2*a*mv
    return largo_desplegado/largo_plegado(a,b,mv,alpha)

def objective(vars,peso_area,peso_ratio):
    a, b, mv, mh, alpha = vars
    return -area(a, b, mv, mh, alpha)*peso_area+ratio(a,b,mv,alpha)*peso_ratio

bounds = opt.Bounds([20, 20, 5, 5, 30], [100, 100, 100, 100, 85])

#non_linear_constraints1 = {'type': 'ineq', 'fun': square_panel}
non_linear_constraints2 = {'type': 'ineq', 'fun': in_cubesat_ancho}
non_linear_constraints3 = {'type': 'ineq', 'fun': in_cubesat_largo}

x0 = [25,25,15,15,50]

peso_area=5
peso_ratio=5

result = opt.minimize(objective, x0, method='trust-constr', bounds=bounds, constraints=[non_linear_constraints2,non_linear_constraints3], options={'maxiter': 5000}, args=(peso_area, peso_ratio))

if result.success:
    a_opt, b_opt, mv_opt, mh_opt, alpha_opt = result.x
    total_area = area(a_opt, b_opt, mv_opt, mh_opt, alpha_opt)
    print("Solución óptima encontrada:")
    print("a =", a_opt, "b =", b_opt, "mv =", mv_opt, "mh =", mh_opt, "alpha =", alpha_opt)
    print("Valor de la función objetivo:", -result.fun)
    print("Lados horizontal,vertical", aristas(a_opt, b_opt, mv_opt, mh_opt, alpha_opt))
    print("Razón de compresión", ratio(a_opt,b_opt,mv_opt,alpha_opt))
else:
    print("No se encontró una solución óptima.")
    print("Mensaje del resultado", result.message)

print("Evaluación de las restricciones:")
#print("square_panel:", square_panel(result.x))
print("in_cubesat_ancho:", in_cubesat_ancho(result.x))
print("in_cubesat_largo:", in_cubesat_largo(result.x))

mv_fixed = mv_opt
mh_fixed = mh_opt
alpha_fixed = alpha_opt

# Rango de valores para 'a' y 'b'
a_vals = np.linspace(bounds.lb[0], bounds.ub[0], 100)
b_vals = np.linspace(bounds.lb[1], bounds.ub[1], 100)
mv_vals = np.linspace(bounds.lb[2], bounds.ub[2], 100)
mh_vals = np.linspace(bounds.lb[3], bounds.ub[3], 100)
alpha_vals = np.linspace(bounds.lb[4], bounds.ub[4], 100)

# Calcular el área para cada combinación de 'a' y 'b'
area_vals = np.zeros((len(a_vals), len(b_vals)))
for i, a in enumerate(a_vals):
    for j, b in enumerate(b_vals):
        area_vals[i, j] = area(a, b, mv_fixed, mh_fixed, alpha_fixed)

# Crear la primera figura y gráfico de contorno para el área en función de a y b
plt.figure()
plt.contourf(a_vals, b_vals, area_vals, cmap='viridis')
plt.colorbar(label='Área')
plt.xlabel('a')
plt.ylabel('b')
plt.title('Área en función de a y b')

# Calcular el área para cada combinación de 'ms' y 'mg'
area_vals2 = np.zeros((len(mv_vals), len(mh_vals)))
for i, ms in enumerate(mv_vals):
    for j, mg in enumerate(mh_vals):
        area_vals2[i, j] = area(a_opt, b_opt, ms, mg, alpha_fixed)

# Crear la segunda figura y gráfico de contorno para el área en función de ms y mg
plt.figure()
plt.contourf(mv_vals, mh_vals, area_vals2, cmap='viridis')
plt.colorbar(label='Área')
plt.xlabel('ms')
plt.ylabel('mg')
plt.title('Área en función de ms y mg')

# Calcular el área para cada combinación de 'alpha' y 'ms'
area_vals3 = np.zeros((len(alpha_vals), len(mv_vals)))
for i, alpha in enumerate(alpha_vals):
    for j, ms in enumerate(mv_vals):
        area_vals3[i, j] = area(a_opt, b_opt, mv_fixed, mh_fixed, alpha)

# Crear la tercera figura y gráfico de contorno para el área en función de alpha y ms
plt.figure()
plt.contourf(alpha_vals, mv_vals, area_vals3, cmap='viridis')
plt.colorbar(label='Área')
plt.xlabel('alpha')
plt.ylabel('ms')
plt.title('Área en función de alpha y ms')

plt.show()
