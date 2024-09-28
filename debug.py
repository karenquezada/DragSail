import sympy as sp

# Definir variables simbólicas
a1, a2 = sp.symbols('a1 a2')
d1, d2 = sp.symbols('d1 d2')

# Definir la función det que calcula el determinante de dos vectores
def det(x, y):
    return x[0]*y[1] - x[1]*y[0]

# Definir la función lineint en sympy
def lineint(a1, d1, a2, d2):
    # Convertir las entradas a vectores
    a1_vector = sp.Matrix(a1)
    a2_vector = sp.Matrix(a2)
    d1_vector = sp.Matrix(d1)
    d2_vector = sp.Matrix(d2)
    
    # Calcular el determinante usando los vectores
    return a1_vector + (d1_vector * (det(a2_vector - a1_vector, d2_vector) / det(d1_vector, d2_vector)))

# p00 = sp.Matrix([1, 0])

# # Definir la variable simbólica x
# x = sp.symbols('x')

# # Definir la función u(x) en sympy
# def u(x):
#     return sp.Matrix([sp.cos(x), sp.sin(x)])

# theta00 = 1.2453160068283862
# phi01 = 0.19811845563178876
# p01 = sp.Matrix([0.5, 0.8660254])
# puntounocero = lineint(p00, u(theta00), p01, u(phi01))
# print(puntounocero)
# Variables para la matriz

# Obtener la matriz simbólica resultante
punto1= sp.symbols('punto1')
punto2= sp.symbols('punto2')
direccion1= sp.symbols('direccion1')
direccion2= sp.symbols('direccion2')

matriz_simbolica = lineint(punto1, direccion1, punto2, direccion2)
print(matriz_simbolica)