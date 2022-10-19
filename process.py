from multiprocessing import Process, Queue
import math
import numpy as np
import time
import multiprocessing

Matrix_A = []
Matrix_B = []
Matrix_C = []

size_of_vectors_n = [int(math.pow(10, 2)), int(math.pow(10, 3)), int(math.pow(10, 4))]
num_of_processes = 1
dimension_N = 500
dimension_M = 30


def Take_input():
    global num_of_processes

    num_of_processes = int(input("Enter the number of processes to run : "))


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


def Matrix_multiply_parallel(matrixa, matrixb, result_Queue):
    temp = 0
    for i in range(0, dimension_N):
        for j in range(dimension_N):
            for k in range(dimension_M):
                temp += float(matrixa[i][k] * matrixb[k][j])
            temp=0

def Multiprocess_function(result_Queue):
    global num_of_processes
    global matrix_dimensions_row
    global Training_data
    global Testing_data

    multiprocess_handle = []
    print(num_of_processes)
    for j in range(0, num_of_processes):
        t = Process(target=Matrix_multiply_parallel,
                    args=(Matrix_A, Matrix_B, result_Queue))

        multiprocess_handle.append(t)
        t.start()

    for k in range(0, num_of_processes):
        multiprocess_handle[k].join()


if __name__ == "__main__":
    Take_input()

    Initialize_Matrix()

    Final_minimum_distance_and_index = multiprocessing.Queue()  # Queue used to share data for multiprocessing in Python
    start_time = time.time()
    Multiprocess_function(Final_minimum_distance_and_index)
    end_time = time.time()
    print("Time taken to multiply two matrices in parallel comes out to be : " + str(
        (end_time - start_time) * 1000) + "ms")