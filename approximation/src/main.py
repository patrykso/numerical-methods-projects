import os
import pandas as pd
import matplotlib.pyplot as plt
import math
import numpy as np

def lagrange(to_evaluate, x_nodes, y_nodes):  
    result = 0
    for i in range(len(x_nodes)):
        tmp = y_nodes[i]
        for j in range(len(x_nodes)):
            if i != j:
                tmp *= (to_evaluate - x_nodes[j]) / (x_nodes[i] - x_nodes[j])
        result += tmp
    return result

def interpolate_using_lagrange(x_nodes, y_nodes, interpolated_count):
    x_interpolated = []
    y_interpolated = []
    
    for i in range(interpolated_count):
        x_interpolated.append(x_nodes[0] + i * (x_nodes[-1] - x_nodes[0]) / interpolated_count)
        y_interpolated.append(lagrange(x_interpolated[-1], x_nodes, y_nodes))
        
    return x_interpolated, y_interpolated

def get_chebyshev_nodes(x, y, nodes):
        x_nodes = [0.5 * (x[0] + x[-1]) + 0.5 * (x[-1] - x[0]) * math.cos((2 * i + 1) * math.pi / (2 * nodes)) for i in range(nodes)]
        y_nodes = []
            
        for node in x_nodes:
            closest = 0
            for i in range(len(x)):
                if abs(x[i] - node) < abs(x[closest] - node):
                    closest = i
            y_nodes.append(y[closest])
            x_nodes[x_nodes.index(node)] = x[closest]
            
        return x_nodes, y_nodes
    
def get_evenly_spaced_nodes(x, y, nodes):
    x_nodes = [x[i] for i in range(0, len(x), len(x)//nodes)]
    y_nodes = [y[i] for i in range(0, len(y), len(y)//nodes)]
    
    return x_nodes, y_nodes

def plot_interpolation(x, y, x_interpolated, y_interpolated, file, nodes, method, x_nodes, y_nodes): 
    plt.title("Interpolacja trasy " + file + "\n dla " + str(nodes) + " węzłów" + " metodą  " + method)
    plt.plot(x, y, label="Dane")
    plt.plot(x_interpolated, y_interpolated, label="Funkcja interpolująca")
    plt.scatter(x_nodes, y_nodes, color="green", label="Węzły")
    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.legend()
    plt.savefig(f"plots/{file}_{nodes}_{method}.png")
    plt.show()

import numpy as np

def splines(x_nodes, interpolated_count, a, b, c, d):
    n = len(x_nodes)
    x_interpolated = []
    y_interpolated = []
    for i in range(interpolated_count):
        x = x_nodes[0] + i * (x_nodes[-1] - x_nodes[0]) / (interpolated_count - 1)
        x_interpolated.append(x)
        
        j = 0
        while j < n - 1 and x_nodes[j + 1] < x:
            j += 1

        dx = x - x_nodes[j]
        y = a[j] + b[j] * dx + c[j] * dx**2 + d[j] * dx**3
        y_interpolated.append(y)
        
    return x_interpolated, y_interpolated

def interpolate_using_splines(x_nodes, y_nodes, interpolated_count):
    n = len(x_nodes)
    V = np.zeros((n, n))
    b = np.zeros(n)
    b_c = np.zeros(n-1)
    a = np.zeros(n-1)
    d = np.zeros(n-1)
    h = x_nodes[1] - x_nodes[0]
    
    # krance
    V[0, 0] = 1
    V[n-1, n-1] = 1
    b[0] = 0
    b[n-1] = 0
    
    # wnetrze
    for i in range(1, n - 1):
        V[i, i - 1] = h
        V[i, i] = 4 * h
        V[i, i + 1] = h
        b[i] = 3 * ((y_nodes[i + 1] - y_nodes[i]) / h - (y_nodes[i] - y_nodes[i - 1]) / h)
        
    c = np.linalg.solve(V, b)
    
    for i in range(n-1):
        a[i] = y_nodes[i]
        b_c[i] = (y_nodes[i + 1] - y_nodes[i]) / h - h * (2 * c[i] + c[i + 1]) / 3
        d[i] = (c[i + 1] - c[i]) / (3 * h)
    
    x_interpolated, y_interpolated = splines(x_nodes, interpolated_count, a, b_c, c, d)
    return x_interpolated, y_interpolated

if __name__ == "__main__":
    for file in os.listdir("profile"):
        print(file)
        data = pd.read_csv(f"profile/{file}")
        x, y = data.iloc[:, 0].tolist(), data.iloc[:, 1].tolist()

        interpolated_count = 777
        nodes_count = [5, 15, 25, 45]
        
        # lagrange interpolation using evenly spaced nodes
        for nodes in nodes_count:
            x_nodes, y_nodes = get_evenly_spaced_nodes(x, y, nodes)
            x_interpolated, y_interpolated = interpolate_using_lagrange(x_nodes, y_nodes, interpolated_count)
            plot_interpolation(x, y, x_interpolated, y_interpolated, file, nodes, "Lagrange (równomiernie rozmieszczone węzły)", x_nodes, y_nodes)
            
        # lagrange interpolation using Chebyshev nodes
        for nodes in nodes_count:
            x_nodes, y_nodes = get_chebyshev_nodes(x, y, nodes)
            x_interpolated, y_interpolated = interpolate_using_lagrange(x_nodes, y_nodes, interpolated_count)
            plot_interpolation(x, y, x_interpolated, y_interpolated, file, nodes, "Lagrange (węzły Czebyszewa)", x_nodes, y_nodes)
            
        # spline interpolation
        for nodes in nodes_count:
            x_nodes, y_nodes = get_evenly_spaced_nodes(x, y, nodes)
            x_interpolated, y_interpolated = interpolate_using_splines(x_nodes, y_nodes, interpolated_count)
            plot_interpolation(x, y, x_interpolated, y_interpolated, file, nodes, "funkcji sklejanych 3. stopnia", x_nodes, y_nodes)
            
        
        