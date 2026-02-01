import numpy as np

def ensure_hermitian(matrix, atol=1e-8):
    if matrix is None:
        raise ValueError("ensure_hermitian received None")

    matrix = np.asarray(matrix)

    if not np.allclose(matrix, matrix.conj().T, atol=atol):
        matrix = 0.5 * (matrix + matrix.conj().T)

    return matrix

