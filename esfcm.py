import numpy as np
from fcm import init_membership_matrix

def init_cluster_centers(X, U_supervised, c):
    N, D = X.shape
    V_new = np.zeros((c, D))
    for i in range(c):
        mat = U_supervised[:,i].reshape((N,1))*X
        numer = np.sum(mat, axis=0)
        denom = np.sum(U_supervised[:,i])
        V_new[i] = numer/denom
    return V_new

def calculate_membership_matrix(X, U_supervised, c, V, lambda_):
    N = X.shape[0]
    U_new = np.zeros((N, c))
    for k in range(N):
        x = X[k]
        distance = np.linalg.norm(x - V, axis=1)**2
        # print(f"x={x}")
        # print(f"v={V}")
        # print(f"d= {distance}")
        numer = np.exp(-lambda_*distance)
        # print(f"tu= {numer}")
        denom = np.sum(numer)
        # print(f"mau= {denom}")
        if denom == 0:
            denom = 1e-10
        summation = np.sum(U_supervised[k])
        for i in range(c):
            U_new[k][i] = U_supervised[k][i] + (numer[i]/denom)*(1 - summation)
    return U_new

def calculate_cluster_centers(X, U, c):
    N, D = X.shape
    V_new = np.zeros((c, D))
    for i in range(c):
        membership = U[:,i].reshape((N, 1))
        numer = np.sum(membership*X, axis=0)
        denom = np.sum(membership)
        V_new[i] = numer/denom
    return V_new

def esfcm(X, c, U_supervised, lambda_, max_iter, eps, init=True):
    N = X.shape[0]
    if init:
        V = init_cluster_centers(X, U_supervised, c)
    else:
        V = init_cluster_centers(X, init_membership_matrix(N,c), c)
    # print(f"V:\n {V}")
    errmax = float('inf')
    i = 0
    while errmax >= eps:
        if i == max_iter:
            break
        V_prev = V
        U = calculate_membership_matrix(X, U_supervised, c, V, lambda_)
        # print(f"U:\n {U}")
        V = calculate_cluster_centers(X, U, c)
        # print(f"V:\n {V}")
        errmax = np.max(np.abs(V - V_prev))
    return U, V


