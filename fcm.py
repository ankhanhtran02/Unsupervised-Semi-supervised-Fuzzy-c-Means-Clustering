import numpy as np
import random

def init_membership_matrix(N, c, upper_bound=1):
    U = []
    for _ in range(N):
        random_num_list = [random.random() for k in range(c)]
        summation = sum(random_num_list)
        generated_list = [(x/summation)*upper_bound for x in random_num_list]
        U.append(generated_list)
    U = np.array(U)
    return U

def calculate_cluster_centers(U, X, c, m):
    """
    Calculate the cluster centers for the Fuzzy C-Means algorithm.
    
    Parameters:
    - U (numpy.ndarray): Membership matrix of shape (N, c)
    - X (numpy.ndarray): Data points matrix of shape (N, D)
    - c (int): Number of clusters
    - m (float): Fuzzifier parameter, typically > 1
    
    Returns:
    - V (numpy.ndarray): Cluster centers of shape (c, D)
    """
    N, D = X.shape
    U_m = U ** m
    V = (U_m.T @ X) / np.sum(U_m, axis=0)[:, np.newaxis]
    return V

def update_membership_matrix(V, X, c, m):
    N, D = X.shape
    U_new = np.zeros((N, c))
    p = 2/(m-1)
    for k in range(N):
        x = X[k] 
        distance = np.linalg.norm(x - V, axis=1)
        for i in range(c):
            denom = sum((distance[i]/distance[j])**p for j in range(c))
            U_new[k][i] = 1/denom
    return U_new

def fcm(X, c, m, max_iter, eps):
    N, D = X.shape
    U = init_membership_matrix(N, c)
    V = np.zeros((c, D))
    errmax = float('inf')
    iter = 0
    while errmax >= eps:
        if iter == max_iter:
            break
        V = calculate_cluster_centers(U, X, c, m)
        U_new = update_membership_matrix(V, X, c, m)
        iter += 1
        errmax = np.max(np.abs(U_new - U))
        U = U_new
    return U, V
