import time
import math
import matplotlib.pyplot as plt
from copy import copy, deepcopy

def create_LU_matrices(matrix):
    U = deepcopy(matrix)
    L = create_matrix(len(matrix))
    for i in range(2, len(matrix) + 1):
        for j in range(1, i):
            L[i-1][j-1] = U[i-1][j-1] / U[j-1][j-1]
            for k in range(len(U[i-1])):
                U[i-1][k] = U[i-1][k] - L[i-1][j-1] * U[j-1][k]
    return L, U                                          

def LU_factorization(matrix, b, x):
    L, U = create_LU_matrices(matrix)
    
    y = create_vector(len(b))
    for i in range(len(matrix)):
        tmp = 0
        for j in range(i):
            tmp += L[i][j] * y[j]
        y[i] = b[i] - tmp
        
    for i in range(len(matrix) - 1, -1, -1):
        tmp = 0
        for j in range(i + 1, len(matrix)):
            tmp += U[i][j] * x[j]
        x[i] = (y[i] - tmp) / U[i][i]
    
    norm = residuum_norm(matrix, b, x)
    return x, norm

def residuum_norm(matrix, b, x):
    residuum = create_vector(len(b))
    for i in range(len(matrix)):
        row_sum = 0
        for j in range(len(matrix)):
            row_sum += matrix[i][j] * x[j]
        residuum[i] = row_sum - b[i]
        
    residuum_sum = 0
    for i in range(len(residuum)):
        try:
            residuum_sum += residuum[i] ** 2
        except OverflowError:
            print("NIE ZBIEGA SIE")
            print(residuum[i])
            print(residuum_sum)
            print(math.sqrt(residuum_sum))
            print(i)
            print("NIE ZBIEGA SIE")
            return 0
    return math.sqrt(residuum_sum)

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
    norm = residuum_norm(matrix, b, x)
    while norm > 10E-10:
        if type == "jacobi":
            x = jacobi_iteration(matrix, b, x)
        else: # gauss-seidel
            x = gauss_iteration(matrix, b, x)
        norm = residuum_norm(matrix, b, x)
        norms.append(norm)
        iterations += 1
    return x, iterations, norms

def matrix_fill(matrix, a1):
    for i in range(len(matrix)):
        for j in range(len(matrix)):
            if i == j:
                matrix[i][j] = a1
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
    print("PROGRAM START")
    
    # ZADANIE A
    matrix = create_matrix(int("9" + str(album)[4] + str(album)[5]))
    matrix = matrix_fill(matrix, 5 + int(str(album)[3]))
    b = create_vector(int("9" + str(album)[4] + str(album)[5]))
    b = b_vector_fill(b, int(str(album)[2]))
    x = create_vector(int("9" + str(album)[4] + str(album)[5]))
    
    # ZADANIE B
    # jacobi
    start_timer = time.time()
    jacobi_x, iterations, norms = iterative("jacobi", deepcopy(matrix), deepcopy(b), deepcopy(x))
    end_timer = time.time()
    print("Jacobi time: ", end_timer - start_timer)
    print("Jacobi iterations: ", iterations, "error: ", norms[-1], "\n")
    
    plt.plot(norms, label="Jacobi")
    plt.yscale('log')
    plt.xlabel('Iteracja')
    plt.ylabel('Norma residuum')
    plt.title("Porównanie norm residuum dla metod iteracyjnych")
    
    # gauss-seidel
    start_timer = time.time()
    gauss_x, iterations, norms = iterative("gauss", deepcopy(matrix), deepcopy(b), deepcopy(x))
    end_timer = time.time()
    print("Gauss time: ", end_timer - start_timer)
    print("Gauss iterations: ", iterations, "error: ", norms[-1], "\n")
    
    plt.plot(norms, label="Gauss-Seidel")
    plt.legend()
    plt.savefig("zadB.png")
    plt.show()
    
    # ZADANIE 3
    matrix = matrix_fill(matrix, 3)
    
    # jacobi
    start_timer = time.time()
    jacobi_x, iterations, norms = iterative("jacobi", deepcopy(matrix), deepcopy(b), deepcopy(x))
    end_timer = time.time()
    print("Jacobi time: ", end_timer - start_timer)
    print("Jacobi iterations: ", iterations, "error: ", norms[-1], "\n")
    
    plt.plot(norms, label="Jacobi")
    plt.yscale('log')
    plt.xlabel('Iteracja')
    plt.ylabel('Norma residuum')
    plt.title("Porównanie norm residuum dla metod iteracyjnych")
    
    # gauss-seidel
    start_timer = time.time()
    gauss_x, iterations, norms = iterative("gauss", deepcopy(matrix), deepcopy(b), deepcopy(x))
    end_timer = time.time()
    print("Gauss time: ", end_timer - start_timer)
    print("Gauss iterations: ", iterations, "error: ", norms[-1], "\n")
    
    plt.plot(norms, label="Gauss-Seidel")
    plt.legend()
    plt.savefig("zadC1.png")
    plt.ylim(10e-2, 300)
    plt.xlim(0, 100)
    plt.savefig("zadC2.png")
    plt.show()

    # ZADANIE D
    # lu factorization
    matrix_fill(matrix, int(str(album)[3]))
    start_timer = time.time()
    x, norms = LU_factorization(deepcopy(matrix), deepcopy(b), deepcopy(x))
    end_timer = time.time()
    print("LU factorization time: ", end_timer - start_timer)
    print("LU factorization error: ", norms)
    
    # ZADANIE E
    matrices_sizes = [100, 500, 1000, 2000, 3000, 4000]
    # matrices_sizes = [100, 200, 300, 500] # debug
    time_taken_jacobi = []
    time_taken_gauss_seidel = []
    time_taken_lu_factorization = []
    
    for i in range(len(matrices_sizes)):
        matrix = create_matrix(matrices_sizes[i])
        matrix = matrix_fill(matrix, 5 + int(str(album)[3]))
        b = create_vector(matrices_sizes[i])
        b = b_vector_fill(b, int(str(album)[2]))
        x = create_vector(matrices_sizes[i])
        
        start_timer = time.time()
        jacobi_x, iterations, norms = iterative("jacobi", deepcopy(matrix), deepcopy(b), deepcopy(x))
        end_timer = time.time()
        print("Jacobi time: ", end_timer - start_timer, "size: ", matrices_sizes[i])
        print("Jacobi iterations: ", iterations, "error: ", norms[-1], "\n")
        time_taken_jacobi.append(end_timer - start_timer)
        
        start_timer = time.time()
        gauss_x, iterations, norms = iterative("gauss", deepcopy(matrix), deepcopy(b), deepcopy(x))
        end_timer = time.time()
        print("Gauss time: ", end_timer - start_timer, "size: ", matrices_sizes[i])
        print("Gauss iterations: ", iterations, "error: ", norms[-1], "\n")
        time_taken_gauss_seidel.append(end_timer - start_timer)
        
        matrix_fill(matrix, int(str(album)[3]))
        start_timer = time.time()
        x, norms = LU_factorization(deepcopy(matrix), deepcopy(b), deepcopy(x))
        end_timer = time.time()
        print("LU factorization time: ", end_timer - start_timer, "size: ", matrices_sizes[i])
        print("LU factorization error: ", norms, "\n")
        time_taken_lu_factorization.append(end_timer - start_timer)
        
    plt.plot(matrices_sizes, time_taken_jacobi, label="Jacobi")
    plt.plot(matrices_sizes, time_taken_gauss_seidel, label="Gauss-Seidel")
    plt.plot(matrices_sizes, time_taken_lu_factorization, label="LU factorization")
    plt.xlabel('Rozmiar macierzy')
    plt.ylabel('Czas [s]')
    plt.title("Porównanie czasów wykonania dla różnych rozmiarów macierzy")
    plt.legend()
    plt.savefig("zadE.png")
    plt.show()
        
        
