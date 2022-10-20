# -*- coding: utf-8 -*-

import math
import numpy as np
import time

Matrix_A = []
Matrix_B = []
Matrix_C = []

size_of_vectors_n = [int(math.pow(10, 2)), int(math.pow(10, 3)), int(math.pow(10, 4))]

dimension_N = 50
dimension_M = 30
num_of_threads = 1

np.set_printoptions(suppress=True)


def Input_for_matrix_dimensions():
    global dimension_N  # matrix_A column and matrix B row
    global dimension_M  # matrix_A row and matrix B column
    global num_of_threads  # threads num


def funA(i, j):
    return (i + 1) * (j + 1) + 0.58


def funB(i, j):
    return (i + 1) * (j + 1) + 0.47


def Initialize_Matrix():
    global Matrix_A
    global Matrix_B
    global Matrix_C
    # for i in range(0,dimension_N):
    #    Matrix_A.append([random.randint(1,5) for i in range(0,dimension_N)])
    # print(Matrix_A)

    # Using numpy to generate matrix
    Matrix_A = np.fromfunction(funA, (dimension_N, dimension_M))
    Matrix_A = Matrix_A.astype(float)

    Matrix_B = np.fromfunction(funB, (dimension_M, dimension_N))
    Matrix_B = Matrix_B.astype(float)

    Matrix_C = np.zeros((dimension_N, dimension_N))
    Matrix_C = Matrix_C.astype(float)


def Matrix_multiply(start, end):
    for i in range(start, end):
        for j in range(dimension_N):
            for k in range(dimension_M):
                Matrix_C[i][j] += float(Matrix_A[i][k] * Matrix_B[k][j])


if __name__ == "__main__":
    Input_for_matrix_dimensions()
    Initialize_Matrix()

    start_time = time.time()
    Matrix_multiply(0, 50)
    end_time = time.time()

    print("The result of a for-loop multiplication of two matrices: " + str(
        (end_time - start_time) * 1000) + "ms")
