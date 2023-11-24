import numpy as np
from eigen_solver import power_iteration_method
from eigen_solver import jacobi_rotations_method
import time

# Проверка метода степенной итерации вместе с построением графика сходимости:

matrix1 = np.random.randint(low=0, high=20, size=(5, 5))

print(f"Max eigenvalue calculated with np.linalg.eigvals function is {max(np.linalg.eigvals(matrix1))}")

print(f"Max eigenvalue is {power_iteration_method(matrix1, plot=True)[0]}\n")

# Проверка метода вращений(только для симметричных)

matrix2 = np.random.randint(low=0, high=20, size=(3, 3))
matrix2 = (matrix2 + matrix2.T)/2

print(f"Our eigenvalues: {jacobi_rotations_method(matrix2)[0]}")

print(f"Eigenvalues calculated with np.linalg.eigvals function is {np.linalg.eigvals(matrix2)}")
