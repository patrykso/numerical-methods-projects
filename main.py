import time
import math
from copy import deepcopy

def LU(matrix, b, x):
    pass

def residuum_norm(matrix, b, x):
    #TODO RESIDUUM NORM
    pass

def gauss_iteration(matrix, b, x):
    for i in range(len(matrix)):
        row_sum = 0
        for j in range(len(matrix)):
            if j != i:
                row_sum += matrix[i][j] * x[j]
        x[i] = (b[i] - row_sum) / matrix[i][i]
    return x
    

def jacobi_iteration(matrix, b, x):
    x_old = deepcopy(x)
    for i in range(len(matrix)):
        row_sum = 0
        for j in range(len(matrix)):
            if j != i:
                row_sum += matrix[i][j] * x_old[j]
        x[i] = (b[i] - row_sum) / matrix[i][i]
    
    return x

def iterative(type, matrix, b, x):
    iterations = 0
    norms = []
    residuum = residuum_norm(matrix, b, x)
    while residuum >= 10 ** -9:
        if type == "jacobi":
            x = jacobi_iteration(matrix, b, x)
        else: # gauss-seidel
            x = gauss_iteration(matrix, b, x)
        residuum = residuum_norm(matrix, b, x)
        norms.append(residuum)
        iterations += 1
    return x, iterations, norms

def matrix_fill(matrix, a1):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j:
                matrix[i][j] = 5 + a1
            elif i == j + 1 or i == j - 1 or i == j + 2 or i == j - 2:
                matrix[i][j] = -1
    return matrix

def print_matrix(matrix):
    for row in matrix:
        print(row)

def create_matrix(n):
    matrix = []
    for i in range(n):
        row = []
        for j in range(n):
            row.append(0)
        matrix.append(row)
    return matrix

def create_vector(n):
    vector = []
    for i in range(n):
        vector.append(0)
    return vector

def b_vector_fill(b, f):
    for i in range(len(b)):
        b[i] = math.sin((i + 1) * (f+1))
    return b

if __name__ == "__main__":
    album = 191711
    
    # matrix = create_matrix(int("9" + str(album)[4] + str(album)[5]))
    matrix = create_matrix(int("9" + str(album)[4] + str(album)[5]) - 901) # debug matrix
    matrix = matrix_fill(matrix, int(str(album)[3]))
    # print_matrix(matrix) # debug
    b = create_vector(int("9" + str(album)[4] + str(album)[5]) - 901) # debug b
    # b = create_vector(int("9" + str(album)[4] + str(album)[5]))
    b = b_vector_fill(b, int(str(album)[2]))
    # print(b) # debug
    x = create_vector(int("9" + str(album)[4] + str(album)[5]) - 901) # debug x
    # x = create_vector(int("9" + str(album)[4] + str(album)[5]))
    # print(x) # debug
    residuum = 10 ** -9
    
    # jacobi
    start_timer = time.time()
    jacobi_x, iterations, norms = iterative("jacobi", matrix, b, deepcopy(x))
    # print(jacobi_x, iterations)
    end_timer = time.time()
    print("Jacobi: ", end_timer - start_timer)
    
    # gauss
    start_timer = time.time()
    gauss_x, iterations, norms = iterative("jacobi", matrix, b, deepcopy(x))
    # print(gauss_x, iterations)
    end_timer = time.time()
    print("Gauss: ", end_timer - start_timer)

    # lu factorization
    start_timer = time.time()
    x, iterations, norms = LU(matrix, b, deepcopy(x))
    end_timer = time.time()
    print("LU factorization: ", end_timer - start_timer)
    
    #TODO WYKRESY