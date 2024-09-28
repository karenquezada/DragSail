import numpy as np

def Rm(k, m):
    angle = 2 * (np.pi / m) * k
    rotation_matrix = np.array([[np.cos(angle), -np.sin(angle)],
                                [np.sin(angle), np.cos(angle)]])
    return rotation_matrix

def det(x, y):
    return x[0] * y[1] - x[1] * y[0]

def lineint(a1, d1, a2, d2):
    return a1 + (d1 * (det((a2 - a1), d2) / det(d1, d2)))

def theta(i, j, delta, alpha, epsilon, m):
    return (np.pi - delta - alpha) + i * epsilon + 2 * np.pi * j / m

def phi(i, j, delta, epsilon, m):
    return (-delta) + i * epsilon + (2 * np.pi * j) / m

def u(x):
    return np.array([np.cos(x), np.sin(x)])

p00 = np.array([1, 0])

def p(i, j, orden, delta, alpha, epsilon):
    # j wrap around 
    
    j = j % orden
    # if i<0:
    #     #print("i", i, "j", j,"p00\n")
    #     assert False
    if j == 0:
        if i == 0:
            #print("i", i, "j", j,"p00\n")
            return p00
        else:
            #print("i", i, "j", j,"lineint")
            return lineint(p(i-1, 0, orden, delta, alpha, epsilon), u(theta(i-1, 0, delta, alpha, epsilon, orden)), 
                           p(i-1, 1, orden, delta, alpha, epsilon), u(phi(i-1, 1, delta, epsilon, orden)))
    else:
        #print("i", i, "j", j,"Rm")
        return Rm(j, orden) @ p(i, 0, orden, delta, alpha, epsilon)

def dk(h, epsilon, m):
    # distancia a la cual hay que ubicar el vértice de la V que comienza un pliegue reverso
    return h * np.cos(epsilon / 2) * (np.sqrt(1 / np.sin(np.pi / m) * 1 / np.sin(np.pi / m - epsilon)))

def module_p(i, j, orden, delta, alpha, epsilon):
    p_ij = p(i, j, orden, delta, alpha, epsilon)
    module = np.linalg.norm(p_ij)
    return module

def e_distance(p1, p2):
    return np.linalg.norm(p1 - p2)

def rho(i, j, k, delta, alpha, eta, epsilon, m):
    return theta(i, j, delta, alpha, epsilon, m) + eta + k * epsilon

def sigma(i, j, k, delta, alpha, eta, epsilon, m):
    return theta(i, j, delta, alpha, epsilon, m) - eta + k * epsilon

def r_orden(i, j, k, orden, delta, alpha, eta, epsilon): 
    j = j % orden
    if i % orden == 0:
        if k == 0:
            return p(i, j, orden, delta, alpha, epsilon)
        else:
            return lineint(r_orden(i, j, k-1, orden, delta, alpha, eta, epsilon), u(rho(i, j, k-1, delta, alpha, eta, epsilon, orden)),
                           p(i+k-1, j+1, orden, delta, alpha, epsilon), u(phi(i+k-1, j+1, delta, epsilon)))

def s_orden(i, j, k, orden, delta, alpha, eta, epsilon): 
    j = j % orden
    if k == 0:
        return p(i * orden, j, orden, delta, alpha, epsilon)
    else:
        return lineint(s_orden(i, j, k-1, orden, delta, alpha, eta, epsilon), u(sigma(i, j, k-1, delta, alpha, eta, epsilon, orden)),
                       p(i + k, j, orden, delta, alpha, epsilon), u(phi(i + k, j, delta, epsilon)))

def calculate_centroid(points):
    return np.mean(points, axis=0)
