import numpy as np
import matplotlib.pyplot as plt

def power_iteration_method(matrix: np.ndarray, plot: bool = False, num_iterations: int = 1000, epsilon: float = 1e-8) -> tuple[float, np.ndarray]:

    if matrix.shape[0] != matrix.shape[1]:
        raise("Incorrect shape of matrix")

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
