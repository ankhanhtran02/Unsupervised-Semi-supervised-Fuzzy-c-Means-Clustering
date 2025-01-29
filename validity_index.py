import numpy as np

def calculate_xie_beni_index(U, V, X):
    """
    Calculate the Xie-Beni index for clustering.
    
    Parameters:
    - U: np.ndarray, membership matrix of shape (N, c) where N is the number of data points and c is the number of clusters.
    - V: np.ndarray, array of cluster centers of shape (c, p), where p is the dimensionality of the data.
    - X: np.ndarray, data points of shape (N, p).
    
    Returns:
    - xb_index: float, the Xie-Beni index value.
    """
    N, c = U.shape
    
    distances = np.zeros((N, c))
    for i in range(c):
        distances[:, i] = np.linalg.norm(X - V[i], axis=1) ** 2

    sigma = np.sum((U ** 2) * distances)

    min_distance = np.inf
    for i in range(c):
        for j in range(i + 1, c):
            distance_between_centers = np.linalg.norm(V[i] - V[j]) ** 2
            if distance_between_centers < min_distance:
                min_distance = distance_between_centers

    if min_distance == 0:
        # min_distance = 1e-10
        raise ValueError("Cluster centers are overlapping; Xie-Beni index is undefined.")

    xb_index = sigma / (N * min_distance)
    return xb_index

def calculate_partition_coefficient(U):
    """
    Calculate partition coefficient (v_PC).

    Parameters:
    - U: np.ndarray, membership matrix of shape (N, c), 
         where c is the number of clusters and n is the number of data points.

    Returns:
    - v_PC: float, partition coefficient value.
    """
    N = U.shape[0] 
    v_PC = np.sum(U ** 2) / N
    return v_PC


def calculate_partition_entropy(U, base=np.e):
    """
    Calculate partition entropy (v_PE).

    Parameters:
    - U: np.ndarray, membership matrix of shape (N, c), 
         where c is the number of clusters and n is the number of data points.
    - base: float, base of the logarithm (default: natural log, base e).

    Returns:
    - v_PE: float, partition entropy value.
    """
    n = U.shape[0]

    U = np.where(U == 0, 1e-10, U)
    v_PE = -np.sum(U * np.log(U) / np.log(base)) / n
    return v_PE


def calculate_bonding_matrix(U):
    '''
    Parameter:
    - U: np.ndarray, membership matrix of shape (n, c) where c is the number of clusters and n is the number of data points.
    '''
    row_norms = np.linalg.norm(U, axis=1, keepdims=True)
    U_normalized = U / row_norms
    return np.dot(U_normalized, U_normalized.T)

def f(X: np.ndarray):
    '''
    Parameter:
    - X: np.ndarray, a square array.
    '''
    return np.sum(X)/2

def g(X: np.ndarray):
    '''
    Parameter:
    - X: np.ndarray, a square array.
    '''
    return f(X) - len(X)/2 


def calculate_index_parameters(B1, B2):
    '''
    Parameters:
    - B1: np.ndarray, bonding matrix of partition 1, shape (n, n), where n is the number of elements.
    - B2: np.ndarray, bonding matrix of partition 2, shape (n, n), where n is the number of elements.
    '''
    a = g(B1 * B2.T)
    b = f((1 - B1) * B2.T)
    c = f(B1 * (1 - B2))
    d = f((1 - B1) * (1 - B2))
    return a, b, c, d

def calculate_external_criteria(U_predicted, U_true):
    '''
    Calculate the fuzzy Rand index, adjusted Rand index and Jaccard index.

    Parameters:
    - U_predicted: np.ndarray, membership matrix of a fuzzy partition, shape (n, c), where n is the number of elements and c is the number of clusters.
    - U_true: np.ndarray, discriminant matrix of a reference hard partition, shape (n, k), where k is the number of classes.
    '''
    B1 = calculate_bonding_matrix(U_predicted)
    B2 = calculate_bonding_matrix(U_true)
    a, b, c, d = calculate_index_parameters(B1, B2)
    numer = 2 * (a * d - b * c)
    denom = c ** 2 + b ** 2 + 2 * a * d + (a + d) * (c + b)
    FARI = numer / denom
    FRI = (a + d) / (a + b + c + d)
    FJI = a / (a + b + c)
    return FRI, FARI, FJI

def one_hot_encode(array, num_classes=None):
    """
    Convert a numpy array of integer labels into one-hot encoding.

    Parameters:
    - array: np.ndarray, array of integer labels.
    - num_classes: int, total number of classes (optional). If not provided, it is inferred from the data.

    Returns:
    - one_hot: np.ndarray, one-hot encoded matrix of shape (len(array), num_classes).
    """
    if num_classes is None:
        num_classes = np.max(array) + 1

    one_hot = np.zeros((len(array), num_classes))

    one_hot[np.arange(len(array)), array] = 1

    return one_hot


