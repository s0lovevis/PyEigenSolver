import numpy as np
from typing import Union
import eigen_solver


def is_eigen_vector(matrix: np.ndarray,
                    v: np.ndarray) -> bool:
    """
    Проверяет, является ли заданный вектор `v` собственным вектором для матрицы `matrix`.

    Args:
    - matrix (np.ndarray): Квадратная матрица, для которой проверяется собственный вектор.
    - v (np.ndarray): Вектор, который проверяется на соответствие собственному вектору.

    Returns:
    - bool: True, если вектор `v` является собственным вектором для матрицы `matrix`, иначе False.

    Raises:
    - ValueError: Если размерности матрицы и вектора не совпадают или вектор является нулевым.
    """
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
                   value: float) -> bool:
    """
    Проверяет, является ли заданное значение `value` собственным значением для матрицы `matrix`.

    Args:
    - matrix (np.ndarray): Квадратная матрица, для которой проверяется собственное значение.
    - value (float): Значение, которое проверяется на соответствие собственному значению.

    Returns:
    - bool: True, если значение `value` является собственным значением для матрицы `matrix`, иначе False.

    Raises:
    - ValueError: Если размерности матрицы не совпадают или собственное число подано нулевым.
    """
    if matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Матрица должна быть квадратной")
    if value == 0:
        raise ValueError("Собственное число не может быть нулевым")

    I = np.eye(matrix.shape[0])
    det = np.linalg.det(matrix - value * I)

    return np.isclose(det, 0, atol=1e-5)


def generate_matrix(eigenvalues: Union[list, np.ndarray], symmetric: bool=False) -> np.ndarray:
    """
    Генерирует квадратную матрицу с заданными собственными значениями.

    Args:
    - eigenvalues (Union[list, np.ndarray]): Список или массив собственных значений.
    - symmetric (bool = False): нужно ли генерировать матрицу симметричной

    Returns:
    - np.ndarray: Квадратная матрица, у которой заданы собственные значения.

    Raises:
    - ValueError: Если вектор собственных значений пуст.
    """
    eigenvalues = np.array(eigenvalues).flatten()

    if len(eigenvalues) == 0:
        raise ValueError("Вектор собственных значений не может быть пустым")

    random_matrix = np.random.randint(low=-10, high=10, size=(len(eigenvalues), len(eigenvalues)))
    if symmetric:
        random_matrix = eigen_solver.qr_decomposition(random_matrix)[0]

    matrix = np.diag(eigenvalues)
    matrix = np.linalg.inv(random_matrix) @ matrix @ random_matrix

    return matrix
