import numpy as np
import scipy.optimize as opt
import matplotlib.pyplot as plt

def largo_plegado(a, b, alpha):
    alpha = np.radians(alpha)
    beta = np.pi - 2 * alpha
    c = np.sqrt(2 * (a**2) + 2 * (a**2) * np.cos(beta))
    return c + b

def largo_desplegado(a):
    return 2 * a 

def objective(vars):
    a, b, alpha = vars
    plegado = largo_plegado(a, b, alpha)
    desplegado = largo_desplegado(a)
    razon_compresion = desplegado / plegado
    return 1 - razon_compresion

bounds = opt.Bounds([20, 20, 30], [100, 100, 85])
x0 = [35, 25, 50]

result = opt.minimize(objective, x0, method='trust-constr', bounds=bounds, options={'maxiter': 5000})

if result.success:
    a_opt, b_opt, alpha_opt = result.x
    print("Solución óptima encontrada:")
    print("a =", a_opt, "b =", b_opt, "alpha =", alpha_opt)
    print("Valor de la función objetivo:", result.fun)

    # Valores óptimos fijos
    a_fixed = a_opt
    b_fixed = b_opt
    alpha_fixed = alpha_opt

    # Rango de valores para 'a', 'b', y 'alpha'
    a_vals = np.linspace(bounds.lb[0], bounds.ub[0], 100)
    b_vals = np.linspace(bounds.lb[1], bounds.ub[1], 100)
    alpha_vals = np.linspace(bounds.lb[2], bounds.ub[2], 100)

    # Calcular el largo plegado para cada combinación de 'a' y 'b'
    largo_vals1 = np.zeros((len(a_vals), len(b_vals)))
    for i, a in enumerate(a_vals):
        for j, b in enumerate(b_vals):
            largo_vals1[i, j] = largo_plegado(a, b, alpha_fixed)

    # Crear el primer gráfico de contorno para el largo en función de a y b
    plt.figure()
    plt.contourf(a_vals, b_vals, largo_vals1, cmap='viridis')
    plt.colorbar(label='Largo Plegado')
    plt.xlabel('a')
    plt.ylabel('b')
    plt.title('Largo Plegado en función de a y b')

    # Calcular el largo plegado para cada combinación de 'alpha' y 'b'
    largo_vals2 = np.zeros((len(alpha_vals), len(b_vals)))
    for i, alpha in enumerate(alpha_vals):
        for j, b in enumerate(b_vals):
            largo_vals2[i, j] = largo_plegado(a_fixed, b, alpha)

    # Crear el segundo gráfico de contorno para el largo en función de alpha y b
    plt.figure()
    plt.contourf(alpha_vals, b_vals, largo_vals2, cmap='viridis')
    plt.colorbar(label='Largo Plegado')
    plt.xlabel('alpha')
    plt.ylabel('b')
    plt.title('Largo Plegado en función de alpha y b')

    # Calcular el largo plegado para cada combinación de 'alpha' y 'a'
    largo_vals3 = np.zeros((len(alpha_vals), len(a_vals)))
    for i, alpha in enumerate(alpha_vals):
        for j, a in enumerate(a_vals):
            largo_vals3[i, j] = largo_plegado(a, b_fixed, alpha)

    # Crear el tercer gráfico de contorno para el largo en función de alpha y a
    plt.figure()
    plt.contourf(alpha_vals, a_vals, largo_vals3, cmap='viridis')
    plt.colorbar(label='Largo Plegado')
    plt.xlabel('alpha')
    plt.ylabel('a')
    plt.title('Largo Plegado en función de alpha y a')

    plt.show()

else:
    print("No se encontró una solución óptima.")
    print("Mensaje del resultado:", result.message)
