import numpy as np
import matplotlib.pyplot as plt
from copy import copy

import numpy as np

def qr_decomposition(matrix):

    n = matrix.shape[0]

    Q = np.zeros((n, n), dtype=float)
    R = np.zeros((n, n), dtype=float)

    for i in range(n):

        v = matrix[:, i].astype(float)

        for j in range(i):
            R[j, i] = np.dot(Q[:, j], matrix[:, i])
            v -= R[j, i] * Q[:, j]

        R[i, i] = np.linalg.norm(v)
        Q[:, i] = v / R[i, i]

    return Q, R


def qr_method(matrix: np.ndarray,
                 num_iterations: int = 1000,
                 epsilon: float = 1e-8) -> np.ndarray:
    A_k = copy(matrix).astype(float)
    for i in range(num_iterations):
        Q, R = qr_decomposition(A_k)
        A_k = R @ Q

        convergence = np.sum(np.abs(A_k - np.diag(np.diagonal(A_k))))
        if convergence < epsilon:
            break

    return np.diagonal(A_k)


def power_iteration_method(matrix: np.ndarray,
                           plot: bool = False,
                           num_iterations: int = 1000,
                           epsilon: float = 1e-8) -> tuple[float, np.ndarray]:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Incorrect shape of matrix")

    vector = np.random.rand(matrix.shape[0])
    vector = vector / np.linalg.norm(vector)

    convergence_history = []

    for i in range(num_iterations):
        new_vector = matrix @ vector
        new_vector = new_vector / np.linalg.norm(new_vector)

        convergence = np.linalg.norm(new_vector - vector)
        convergence_history.append(convergence)

        if convergence < epsilon:
            break

        vector = new_vector

    eigenvalue = np.linalg.norm(matrix @ vector) / np.linalg.norm(vector)

    if plot:
        plt.figure(figsize=(8, 6))
        plt.plot(range(len(convergence_history)), convergence_history, marker='o', linestyle='-')
        plt.xlabel('Итерация')
        plt.ylabel('Сходимость')
        plt.title('Зависимость сходимости от итерации')
        plt.grid(True)

        plt.savefig('convergence_plot.png')
        plt.close()

    return eigenvalue, vector


def jacobi_rotations_method(matrix: np.ndarray,
                            num_iterations: int = 1000,
                            epsilon: float = 1e-8) -> tuple[np.ndarray, np.ndarray]:
    n = len(matrix)

    for i in range(n):
        for j in range(n):
            if matrix[i][j] != matrix.T[i][j]:
                raise ValueError("Matrix is not symmetric")

    cumulative_rot = np.identity(n)

    for _ in range(num_iterations):

        max_off_diag = -1.0
        row, col = 0, 0
        for i in range(n - 1):
            for j in range(i + 1, n):
                if abs(matrix[i, j]) > max_off_diag:
                    max_off_diag = abs(matrix[i, j])
                    row, col = i, j

        if max_off_diag < epsilon:
            break

        a_ii = matrix[row, row]
        a_jj = matrix[col, col]
        a_ij = matrix[row, col]
        if abs(a_ij) < epsilon:
            theta = 0.5 * np.pi if a_ii > a_jj else -0.5 * np.pi
        else:
            if a_ii != a_jj:
                theta = 0.5 * np.arctan(2 * a_ij / (a_ii - a_jj))
            else:
                theta = np.pi/4

        c = np.cos(theta)
        s = np.sin(theta)
        rotation_matrix = np.identity(n)
        rotation_matrix[row, row] = c
        rotation_matrix[row, col] = -s
        rotation_matrix[col, row] = s
        rotation_matrix[col, col] = c

        matrix = rotation_matrix.T @ matrix @ rotation_matrix

    eigenvalues = np.diag(matrix)
    cumulative_rot = cumulative_rot @ rotation_matrix

    return eigenvalues, cumulative_rot
