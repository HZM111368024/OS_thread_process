# -*- coding: utf-8 -*-

from threading import Thread
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

    num_of_threads = int(input("Enter the number of threads : "))


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
    print("Matrix_A : \n" + str(Matrix_A))

    Matrix_B = np.fromfunction(funB, (dimension_M, dimension_N))
    Matrix_B = Matrix_B.astype(float)
    print("Matrix_B : \n" + str(Matrix_B))

    Matrix_C = np.zeros((dimension_N, dimension_N))
    Matrix_C = Matrix_C.astype(float)


def Matrix_multiply_parallel_50(start, end, thread):
    for i in range(start, end):
        for j in range(dimension_N):
            for k in range(dimension_M):
                Matrix_C[i][j] += float(Matrix_A[i][k] * Matrix_B[k][j])


def Matrix_multiply_parallel_10(start, end, col, thread):
    for i in range(start, end):
        for j in range(col):
            for k in range(dimension_M):
                Matrix_C[i][j] += float(Matrix_A[i][k] * Matrix_B[k][j])


def Thread_function():
    global num_of_threads
    thread_handle = []
    if num_of_threads == 10:
        for j in range(0, num_of_threads):
            if j < 5:
                t = Thread(target=Matrix_multiply_parallel_10,
                           args=(int((dimension_N / 5) * j), int((dimension_N / 5) * (j + 1)), 25, int(j)))
            else:
                t = Thread(target=Matrix_multiply_parallel_10,
                           args=(int((dimension_N / 5) * (j - 5)), int((dimension_N / 5) * (j - 4)), 50, int(j)))
    else:
        for j in range(0, num_of_threads):
            t = Thread(target=Matrix_multiply_parallel_50,
                       args=(int((dimension_N / num_of_threads) * j), int((dimension_N / num_of_threads) * (j + 1)),int(j)))

    thread_handle.append(t)
    t.start()
    for j in range(0, num_of_threads):
        thread_handle[j].join()


if __name__ == "__main__":
    Input_for_matrix_dimensions()
    Initialize_Matrix()

    start_time = time.time()
    Thread_function()
    print("Matrix_C : \n" + str(Matrix_C))
    end_time = time.time()

    print("Time taken to multiply two matrices in parallel comes out to be : " + str(
        (end_time - start_time) * 1000) + "ms")
