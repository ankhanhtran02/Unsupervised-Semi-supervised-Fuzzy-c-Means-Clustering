import numpy as np
from esfcm import init_cluster_centers
from fcm import init_membership_matrix
from sklearn.preprocessing import StandardScaler

def calculate_membership_matrix(X, U_supervised, c, V, m):
    N = X.shape[0]
    U_new = np.zeros((N, c))
    if m > 1:
        exponent = 1/(m-1)
        for k in range(N):
            x = X[k]
            distance = np.linalg.norm(x - V, axis=1)**2
            inverse_distance = (1/distance)**exponent
            supervised_sum = np.sum(U_supervised[k])
            denom = np.sum(inverse_distance)
            for i in range(c):
                numer = (1 - supervised_sum)*inverse_distance[i]
                U_new[k][i] = U_supervised[k][i] + numer/denom
    elif m == 1:
        for k in range(N):
            x = X[k]
            distance = np.linalg.norm(x - V, axis=1)**2
            min_cluster = np.argmin(distance)
            supervised_sum = np.sum(U_supervised[k])
            for i in range(c):
                if i == min_cluster:
                    U_new[k][i] = U_supervised[k][i] + 1 - supervised_sum
                else:
                    U_new[k][i] = U_supervised[k][i]
    return U_new

def calculate_cluster_centers(X, U, U_supervised, c, m):
    N, D = X.shape
    V_new = np.zeros((c, D))
    delta_U_m = np.absolute(U - U_supervised)**m
    for i in range(c):
        mat = delta_U_m[:, i].reshape((N, 1))*X
        numer = np.sum(mat, axis=0)
        denom = np.sum(delta_U_m[:, i])
        V_new[i] = numer/denom
    return V_new

def ssfcm(X, c, U_supervised, m, max_iter, eps, init=True):
    N = X.shape[0]
    if init:
        V = init_cluster_centers(X, U_supervised, c)
    else:
        V = init_cluster_centers(X, init_membership_matrix(N, c), c)
    # V = calculate_cluster_centers(X, np.zeros((X.shape[0], c)), U_supervised, c, m)
    errmax = float('inf')
    i = 0
    while errmax >= eps:
        if i == max_iter:
            break
        V_prev = V
        U = calculate_membership_matrix(X, U_supervised, c, V, m)
        V = calculate_cluster_centers(X, U, U_supervised, c, m)
        errmax = np.max(np.abs(V - V_prev))
    return U, V

