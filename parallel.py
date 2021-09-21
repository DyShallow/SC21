# Author: Dylan Shallow
# SC21 student outreach workshop
# This program is intended to demonstrate the performance potential of parallelism for certain tasks

import sys      # accept arguments
from multiprocessing import Process # use multiprocessing to improve throughput
import time # record execution time and provide estimates
import random as rand
import numpy as np

# globals. We group these variables here for easy editing.
# For variables that will only change rarely, it can be easier to place them here than enter them in as arguments every time we run the program.
performance_log_file = 'performance_log.txt'

# simple usage output. TODO(Dylan): more descriptive, e.g. list and describe each argument
def usage():
    print(r'Usage: \npy .\python-filename.py [args]')

def add_matrix(mat1, mat2):
    if not (len(mat1) == len(mat2) and len(mat1[0]) == len(mat2[0])): # validate inputs
        print('Matrices are not of the same size, cannot add them')
        quit()
    result = []
    for inner_1,inner_2 in zip(mat1, mat2): # for each pair of lists in the outer list of each matrix
        inner_result = []
        for i,j in zip(inner_1,inner_2): # might be able to replace with list comprehension
            inner_result.append(i + j)
        result.append(inner_result)
    return result

def add_matrix_numpy(mat1, mat2):
    return np.add(mat1, mat2)

def multiply_matrix(mat1, mat2):
    pass

def multiply_matrix_numpy(mat1, mat2):
    return np.matmul(mat1, mat2)

def transpose(mat):
    result = []
    for _ in range(len(mat[0])):
        result.append([])
    for inner in mat:
        for i,element in zip(range(len(inner)),inner):
            result[i].append(element)
    return result

def transpose_numpy(mat):
    return np.transpose(mat)

def generate_matrix(size_inner, size_outer=1, seed=None):
    rand.seed(seed) # defaults to current system time as seed if seed is None
    generated_matrix = []
    for _ in range(size_outer):
        inner_list = []
        for _ in range(size_inner):
            inner_list.append(rand.randrange(1,1000)) # use numbers from 1 to 999 to be a bit more readable
        generated_matrix.append(inner_list)
    return generated_matrix

def generate_vector(size, seed=None):
    return generate_matrix(size_inner=size, seed=seed)

def add_vector(vec1, vec2): # TODO(Dylan): Maybe make these matrices and check dimensionality so that we avoid confusion of passing 'mat' to 'vec' in the worker prepare step
    return add_matrix(vec1, vec2)

def add_vector_numpy(vec1, vec2):
    return np.add(vec1, vec2)

def multiply_vector(vec1, vec2):
    pass

def multiply_vector_numpy(vec1, vec2):
    return multiply_matrix_numpy(vec1, vec2)

def matrices_equal(mat1, mat2):
    return np.equal(mat1, mat2).all()

def vectors_equal(vec1, vec2):
    return matrices_equal(vec1, vec2)

# split matrix 'mat' into 'count' matrices of approximately equal size
def split_matrix(mat, count):
    pass

def reconstruct_split_matrices(mat,dims):
    pass

def main(argv):
    # parse arguments. If we aren't able to parse them properly, remind the user of proper usage and quit
    if len(argv) not in [3,4]: # check length. We use 'in' syntax in case we want to add optional arguments, in which case the length of the argument list might have more than one valid value
        usage()
        quit()

    # Use function pointers so that we can re-use the code which handles parallelism
    tasks = ['add_vector', 'multiply_vector', 'add_matrix', 'multiply_matrix'] # supported operations
    task_functions = [add_vector, multiply_vector, add_matrix, multiply_matrix] # functions corresponding to tasks
    if(argv[0] in tasks):
        function_pointer = task_functions[tasks.index(argv[0])]
    else:
        print(f'Invalid task specified! {argv[0]}')
        usage()
        quit()

    number_of_processes = int(argv[1])

    size_x = int(argv[2])
    if len(argv) >= 4:
        size_y = int(argv[3])
    else:
        size_y = 1

    # record and output start time for our records. This will be used to calculate total runtime
    start_time = time.time()
    start_time_string = time.strftime("%Y-%m-%dT%H:%M:%S",time.localtime(start_time))
    print(f"Start time: {start_time_string}")
    with open(performance_log_file, 'a') as perf_log: # record things in a log so that we don't need to copy/paste from terminal, etc. This reduces the chance we lose the data accidentally, which is especially important for programs that take a long time, e.g. hours, to run.
        perf_log.write('\n')
        for arg in argv:
            perf_log.write('\n' + arg)
        perf_log.write(f"\nStart time: {start_time_string}")

    # generate items
    mat1 = generate_matrix(size_x, size_y)
    mat2 = generate_vector(size_x, size_y)

    # # we need to have some variables for synchronization in some scenarios
    # needs_synchronization = False
    # if needs_synchronization:
    #     manager = multiprocessing.Manager()
    #     shared_dict = manager.dict()

    # start worker processes to perform the work
    workers = []
    for _ in range(number_of_processes): # underscore indicates we do not care about the iterator variable itself
        workers.append(Process(target=function_pointer, args=(mat1, mat2,))) # prepare workers
    for worker in workers:
        worker.start() # start workers

    # join processes
    for worker in workers:
        worker.join()
    print('All processes joined!') # report all processes joined. If we don't get here, maybe one of the workers got stuck

    # output and record end and elapsed time
    end_time = time.time()
    end_time_string = time.strftime("%Y-%m-%dT%H:%M:%S",time.localtime(end_time))
    print(f"End time: {end_time_string}")

    elapsed_time = end_time - start_time
    print(f"Elapsed time: {elapsed_time}")

    with open(performance_log_file, 'a') as perf_log:
        perf_log.write(f"\nEnd time: {end_time_string}")
        perf_log.write(f"\nElapsed time: {elapsed_time}")
    

if __name__ == "__main__":
   main(sys.argv[1:]) # strip implicit first argument, we only care about user-specified args