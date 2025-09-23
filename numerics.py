import numpy as np
from scipy.linalg import solve

def solution(Npt, X, Nddl):
    return np.array([X[Nddl[No] - 1] if Nddl[No] > 0 else 0 for No in range(1, Npt + 1)])

def num_DoF(codnfr):
    Nddl = []
    Neq  = 0
    Nfix = 0
    for i in codnfr:
        Nddl.append([len(Nddl) + 1, Neq + 1 if i == 0 else Nfix - 1])
        if i == 0:
            Neq += 1
        if i == 1:
            Nfix -= 1
    return np.array(Nddl), Neq, abs(Nfix)

def solve_dynamic_system(M, theta, deltat, C, F, Xold, K):
    GM   = (M - deltat * (1 - theta) * (K + C)) @ Xold + deltat * F
    Iter = M + theta * deltat * (K + C)
    Xnew = np.linalg.solve(Iter, GM)
    return Xnew

def rhs_elem(Airetau, beta, alpha, epsilon, z1, z2, z3, f):
    barycentre = [(z1[0] + z2[0] + z3[0]) / 3, (z1[1] + z2[1] + z3[1]) / 3]
    F = Airetau / 3 * f(barycentre[0], barycentre[1], epsilon, alpha, beta) * np.ones((3, 1))
    return F

def conv_elem(beta, X, Y):
    Y_transpose = np.transpose(Y)
    X_transpose = np.transpose(X)
    first_term = beta[0] / 6 * np.vstack((Y_transpose, Y_transpose, Y_transpose))
    second_term = beta[1] / 6 * np.vstack((X_transpose, X_transpose, X_transpose))
    C = first_term + second_term
    return C

def diff_elem(Airetau, X, Y, epsilon):
    Y_transpose = np.transpose(Y)
    X_transpose = np.transpose(X)
    K = epsilon / (4 * Airetau) * (np.outer(Y, Y_transpose) + np.outer(X, X_transpose))
    return K

def mass_elem(Airetau):
    M = np.eye(3) + np.ones((3, 3))
    M = Airetau / 12 * M
    return M

def explicit_euler(u, dt, dx, dy, alpha, num_steps):
    for k in range(num_steps):
        u_new = u.copy()
        for i in range(1, u.shape[0]-1):
            for j in range(1, u.shape[1]-1):
                laplacian = (u[i+1, j] - 2*u[i, j] + u[i-1, j]) / dx**2 + (u[i, j+1] - 2*u[i, j] + u[i, j-1]) / dy**2
                u_new[i, j] = u[i, j] + alpha * dt * laplacian
        u = u_new.copy()
    return u

def crank_nicolson(u, dt, dx, dy, alpha, num_steps):
    r_x = alpha * dt / (2 * dx**2)
    r_y = alpha * dt / (2 * dy**2)
    A = np.eye(u.size) * (1 + 2*r_x + 2*r_y)
    A += np.eye(u.size, k=1) * (-r_x)
    A += np.eye(u.size, k=-1) * (-r_x)
    A += np.eye(u.size, k=u.shape[1]) * (-r_y)
    A += np.eye(u.size, k=-u.shape[1]) * (-r_y)
    for k in range(num_steps):
        b = u.flatten()
        u_new = solve(A, b)
        u = u_new.reshape(u.shape)
    return u

def triang_params(tau, connec, coord):
    N = connec[tau - 1]
    z1 = coord[N[0] - 1]
    z2 = coord[N[1] - 1]
    z3 = coord[N[2] - 1]
    x1, y1 = z1
    x2, y2 = z2
    x3, y3 = z3
    area = 0.5 * abs(x1 * (y2 - y3) + x2 * (y3 - y1) + x3 * (y1 - y2))
    X = np.zeros(3)
    Y = np.zeros(3)
    X[0] = z3[0] - z2[0]
    X[1] = z1[0] - z3[0]
    X[2] = z2[0] - z1[0]
    Y[0] = z2[1] - z3[1]
    Y[1] = z3[1] - z1[1]
    Y[2] = z1[1] - z2[1]
    return N, z1, z2, z3, area, X, Y

def assemble_global_matrices(Nelem, connec, Nddl, F_elements, M_elements, K_elements, C_elements):
    Nnode = max(Nddl[:, 0])  # Anzahl der globalen Knoten

    M_global = np.zeros((Nnode, Nnode))  # Globale Massenmatrix initialisieren
    K_global = np.zeros((Nnode, Nnode))  # Globale Steifigkeitsmatrix initialisieren
    C_global = np.zeros((Nnode, Nnode))  # Globale Konvektionsmatrix initialisieren
    F_global = np.zeros((Nnode, 1))      # Globaler Vektor initialisieren

    for T in range(1, Nelem + 1):
        # Schleife über alle Elemente

        for k in range(3):
            # Schleife über die Knoten des Elements
            Nk = connec[k, T]
            i = Nddl[Nk - 1, 0] - 1  # Index des globalen Knotens

            if i >= 0:
                F_global[i] += F_elements[k, T - 1]

                for l in range(3):
                    # Schleife über die Knoten des Elements für Matrixassemblage
                    Nl = connec[l, T]
                    j = Nddl[Nl - 1, 0] - 1  # Index des globalen Knotens

                    if j >= 0:
                        M_global[i, j] += M_elements[k, l, T - 1]
                        K_global[i, j] += K_elements[k, l, T - 1]
                        C_global[i, j] += C_elements[k, l, T - 1]

    return M_global, K_global, C_global, F_global