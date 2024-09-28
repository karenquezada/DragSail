import numpy as np
import matplotlib.pyplot as plt

def miura_drawer(w, h, alpha, nv, nh):
    angulo = np.deg2rad(alpha)
    d = h / np.tan(alpha)
    
    # Redondear nv y nh al 0.5 más cercano
    nv = round(nv * 2) / 2
    nh = round(nh * 2) / 2
    
    # Dibuja los módulos completos
    for i in range(int(nh)):
        for j in range(int(nv)):
            x_mod = i * 2 * w
            y_mod = j * 2 * h
            A = np.array([x_mod, y_mod])
            B = np.array([x_mod + w, y_mod])
            C = np.array([x_mod + 2 * w, y_mod])
            D = np.array([x_mod, y_mod + 2 * h])
            E = np.array([x_mod + d, y_mod + h])
            F = np.array([x_mod + w, y_mod + 2 * h])
            G = np.array([x_mod + 2 * w, y_mod + 2 * h])
            H = np.array([x_mod + d + 2 * w, y_mod + h])
            I = np.array([x_mod + d + w, y_mod + h])
            
            plt.plot([A[0], B[0]], [A[1], B[1]], 'b-')
            plt.plot([A[0], E[0]], [A[1], E[1]], 'b-')
            plt.plot([B[0], I[0]], [B[1], I[1]], 'b-')
            plt.plot([I[0], E[0]], [I[1], E[1]], 'b-')
            plt.plot([E[0], D[0]], [E[1], D[1]], 'b-')
            plt.plot([D[0], F[0]], [D[1], F[1]], 'b-')
            plt.plot([I[0], F[0]], [I[1], F[1]], 'b-')
            plt.plot([F[0], G[0]], [F[1], G[1]], 'b-')
            plt.plot([G[0], H[0]], [G[1], H[1]], 'b-')
            plt.plot([H[0], I[0]], [H[1], I[1]], 'b-')
            plt.plot([H[0], C[0]], [H[1], C[1]], 'b-')
            plt.plot([C[0], B[0]], [C[1], B[1]], 'b-')
    
    # Dibuja medio módulo vertical
    if nv % 1 != 0:
        for j in range(int(nh)):
            xv = j * 2 * w
            yv = int(nv) * 2 * h
            AV = np.array([xv, yv])
            CV = np.array([xv + 2 * w, yv])
            HV = np.array([xv + d + 2*w, yv + h])
            BV = np.array([xv + w, yv])
            EV = np.array([xv + d, yv + h])
            IV = np.array([xv + d + w, yv + h])
            
            plt.plot([AV[0], BV[0]], [AV[1], BV[1]], 'b-')
            plt.plot([AV[0], EV[0]], [AV[1], EV[1]], 'b-')
            plt.plot([BV[0], IV[0]], [BV[1], IV[1]], 'b-')
            plt.plot([IV[0], EV[0]], [IV[1], EV[1]], 'b-')
            plt.plot([IV[0], HV[0]], [IV[1], HV[1]], 'b-')
            plt.plot([HV[0], CV[0]], [HV[1], CV[1]], 'b-')
            plt.plot([BV[0], CV[0]], [BV[1], CV[1]], 'b-')
    # Dibuja medio módulo horizontal si nh no es un entero
    if nh % 1 != 0:
        for i in range(int(nv)):
            xh = int(nh) * 2 * w
            yh = i * 2 * h
            AH = np.array([xh, yh])
            BH = np.array([xh + w, yh])
            EH = np.array([xh + d, yh + h])
            IH = np.array([xh + d + w, yh + h])
            DH= np.array([xh, yh + 2 * h])
            FH = np.array([xh + w, yh + 2 * h])
            
            plt.plot([AH[0], BH[0]], [AH[1], BH[1]], 'b-')
            plt.plot([AH[0], EH[0]], [AH[1], EH[1]], 'b-')
            plt.plot([BH[0], IH[0]], [BH[1], IH[1]], 'b-')
            plt.plot([IH[0], EH[0]], [IH[1], EH[1]], 'b-')
            plt.plot([EH[0], DH[0]], [EH[1], DH[1]], 'b-')
            plt.plot([DH[0], FH[0]], [DH[1], FH[1]], 'b-')
            plt.plot([IH[0], FH[0]], [IH[1], FH[1]], 'b-')

    def largo(theta): 
        w, h, alpha, nv, nh = theta[0], theta[1], theta[2], theta[3], theta[4]
        #return (nh*w*np.sqrt(2*(1-np.cos(np.pi-2*np.deg2rad(alpha)))) + w ) #casoentero
        return (nh*w*np.sqrt(2*(1-np.cos(np.pi-2*(alpha)))) + w*np.cos((alpha))) #caso medio

   
    plt.axis('equal')
    plt.xlim(0, None)
    plt.ylim(0, None)

    # Add title and axis labels
    plt.title('Origami Miura') 
    plt.xlabel('Ancho (mm)')
    plt.ylabel('Alto (mm)')


    plt.show()