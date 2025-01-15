import matplotlib.pyplot as plt

# SATELLITE ORBITAL DECAY
satellite_mass = 4  # Masa del satélite en kg
solar_radio_flux = 66.8  # en SFU
geomagnetic_a_index = 50
starting_height = 500  # Altura inicial en km


# Constantes
Re = 6378000  # Radio de la Tierra en metros
Me = 5.98E+24  # Masa de la Tierra en kg
G = 6.67E-11  # Constante gravitacional universal
pi = 3.1416

# Áreas transversales
areas = {
    "Sin vela (0.035 m²)": 0.035,
    "Con una vela (0.119 m²)": 0.119,
    "Con dos velas (0.203 m²)": 0.203,
}

# Crear el gráfico
plt.figure(figsize=(10, 6))

for label, satellite_area in areas.items():
    T = 0  # Tiempo inicial en días
    dT = 0.1  # Incremento de tiempo en días
    D9 = dT * 3600 * 24  # Incremento de tiempo en segundos
    H2 = starting_height  # Altura inicial en km
    R = Re + H2 * 1000  # Radio orbital inicial en metros
    P = 2 * pi * ((R**3 / (Me * G))**0.5)  # Periodo orbital inicial en segundos

    # Listas para almacenar datos
    tiempos = []
    alturas = []

    while True:
        SH = (900 + 2.5 * (solar_radio_flux - 70) + 1.5 * geomagnetic_a_index) / (27 - 0.012 * (H2 - 200))
        DN = 6E-10 * (2.71828**(-(H2 - 175) / SH))  # Densidad atmosférica
        dP = 3 * pi * satellite_area / satellite_mass * R * DN * D9  # Decremento en el periodo orbital

        # Almacenar datos
        tiempos.append(T / 365)  # Tiempo en años
        alturas.append(H2)  # Altura en km

        if H2 < 180:
            break

        # Actualización de valores
        P -= dP
        T += dT
        R = (G * Me * P**2 / (4 * pi**2))**(1/3)  # Nuevo radio orbital
        H2 = (R - Re) / 1000  # Nueva altura en km

    # Añadir curva al gráfico
    plt.plot(tiempos, alturas, label=label)

# Configuración del gráfico
plt.title("Decaimiento orbital del satélite con diferentes áreas transversales")
plt.xlabel("Tiempo (años)")
plt.ylabel("Altitud (km)")
plt.grid(True)
plt.legend()
plt.xlim(left=0s)
plt.show()
