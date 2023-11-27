import numpy as np


def is_eigen_vector(matrix: np.ndarray,
                    v: np.ndarray,
                    epsilon: float = 1e-8) -> bool:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    if matrix.shape[0] != len(v):
        raise ValueError("Размерность матрицы и вектора должны совпадать")

    if np.allclose(v, np.zeros_like(v)):
        raise ValueError("Вектор не может быть нулевым")

    eigen_value = (np.linalg.norm(v.T.astype(float) @ matrix.astype(float) @ v.astype(float)) /
                   (v.T.astype(float) @ v.astype(float)))

    return is_eigen_value(matrix, eigen_value)


def is_eigen_value(matrix: np.ndarray,
                   value: float,
                   epsilon: float = 1e-8) -> bool:
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матрица должна быть квадратной")

    I = np.eye()
    det = np.linalg.det(matrix - value * I)

    return np.isclose(det, 0, epsilon)
