import numpy as np


def is_eigen_vector(matrix: np.ndarray,
                    vector: np.ndarray,
                    epsilon: float = 1e-8) -> bool:

    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    if matrix.shape[0] != len(vector):
        raise ValueError("Размерность матрицы и вектора должны совпадать")

    if np.allclose(vector, np.zeros_like(vector)):
        raise ValueError("Вектор не может быть нулевым")

    result = matrix.astype(float) @ vector.astype(float)
    eigen_value = np.linalg.norm(matrix.astype(float) @ vector.astype(float)) / np.linalg.norm(vector.astype(float))

    return np.linalg.norm(result - eigen_value * vector.astype(float)) < epsilon
