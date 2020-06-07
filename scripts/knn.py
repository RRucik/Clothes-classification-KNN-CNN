import utils.mnist_reader
import numpy as np
from utils import mnist_reader


def hamming_distance(X, X_train):

    X_train_transposed = np.transpose(X_train)

    distance_matrix = (~X).astype(int) @ X_train_transposed.astype(int) + X.astype(int) @ (
        ~X_train_transposed).astype(int)
    return distance_matrix

def sort_train_labels_knn(Dist, y):

    sorted_dist = Dist.argsort(kind='mergesort')
    return y[sorted_dist]

def p_y_x_knn(y, k):

    M = len(np.unique(y))
    k_nearest_matrix = np.ndarray(shape=(np.shape(y)[0], k), dtype=int)
    probability_matrix = np.ndarray(shape=(np.shape(y)[0], M))
    for i in range(len(k_nearest_matrix)):
        for j in range(len(k_nearest_matrix[0])):
            k_nearest_matrix[i][j] = y[i][j]

        probability_matrix[i] = np.bincount(k_nearest_matrix[i], None, M) / k

    return probability_matrix

def classification_error(p_y_x, y_true):

    error_count = 0
    for i in range(len(p_y_x)):
        max_probability = 0
        max_index = 0
        for j in range(len(p_y_x[i])):
            if p_y_x[i][j] >= max_probability:
                max_probability = p_y_x[i][j]
                max_index = j
        if max_index != y_true[i]:
            error_count += 1

    error_count /= len(y_true)

    return error_count


def model_selection_knn(X_val, X_train, y_val, y_train, k_values):

    errors = np.zeros(shape=np.shape(k_values))
    sorted_by_distance_matrix = sort_train_labels_knn(hamming_distance(X_val, X_train), y_train)
    for i in range(len(k_values)):
        p_y_x_for_this_k = p_y_x_knn(sorted_by_distance_matrix, k_values[i])
        errors[i] = classification_error(p_y_x_for_this_k, y_val)

    best_error = np.amin(errors)
    best_k = k_values[np.argmin(errors)]

    return (best_error, best_k, errors)

def test_knn():

    X_train, y_train = mnist_reader.load_mnist('../data/fashion', kind='train')
    X_test, y_test = mnist_reader.load_mnist('../data/fashion', kind='t10k')
    k_values = [10,20,30,40,50]
    best_error, best_k, errors = model_selection_knn(X_test, X_train, y_test, y_train, k_values)
    print("Model accuracy: " + str(1.0 - best_error) + " K value for lowest error: " + str(best_k))

test_knn()