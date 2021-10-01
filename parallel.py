# Author: Dylan Shallow
# SC21 student outreach workshop
# This program is intended to demonstrate the performance potential of parallelism for certain tasks

import sys      # accept arguments
from multiprocessing import Process # use multiprocessing to improve throughput. TODO switch to threads to reduce memory requirements?
import time # record execution time and provide estimates
import random as rand
import numpy as np
import math # used for some calculations such as floor and log

# globals. We group these variables here for easy editing.
# For variables that will only change rarely, it can be easier to place them here than enter them in as arguments every time we run the program.
performance_log_file = 'performance_log.txt'

# simple usage output. TODO(Dylan): more descriptive, e.g. list and describe each argument
def usage():
    print(r'Usage: \npy .\python-filename.py [args]')

def get_matrix_dims(matrix):
    if len(matrix) == 0:
        return (0,0)
    return (len(matrix[0]),len(matrix))

def get_result_dims(mat1, mat2):
    mat1_inner, mat1_outer = get_matrix_dims(mat1)
    mat2_inner, mat2_outer = get_matrix_dims(mat2)
    if mat1_inner != mat2_outer:
        print("mat1 outer and mat2 inner dimensions must be the same")
        quit()
    return (mat1_outer, mat2_inner)

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

def multiply_matrix_numpy(mat1, mat2):
    return np.matmul(mat1, mat2)

def multiply_matrix(mat1, mat2_T):
    inner1, outer1 = get_matrix_dims(mat1)
    inner2, outer2 = get_matrix_dims(mat2_T)
    result = [ [0 for _ in range(outer2)] for _ in range(outer1)] #[[0] * outer2] * outer1
    for i in range(outer1): # outer results
        for j in range(outer2): # inner results
            for k in range(inner1): # inner1 and inner2 are same
                result[i][j] += mat1[i][k] * mat2_T[j][k]
    return result

# also known as Hadamard product. We do this because mat2 is already transposed
def multiply_matrix_elementwise(mat1, mat2_T):
    result = []
    for inner1, inner2 in zip(mat1,mat2_T):
        inner_result = []
        for element1, element2 in zip(inner1, inner2):
            inner_result.append(element1 * element2)
        result.append(inner_result)
    return result

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
    result = 0
    for element1, element2 in zip(vec1, vec2):
        result += element1 * element2
    return result

def multiply_vector_numpy(vec1, vec2):
    return multiply_matrix_numpy(vec1, vec2)

def matrices_equal(mat1, mat2):
    return np.equal(mat1, mat2).all()

def vectors_equal(vec1, vec2):
    return matrices_equal(vec1, vec2)

# split matrix of shape 'dims' into 'count' matrix dims of approximately equal size. This will represent the dims of the result matrix so they should be square if possible to minimize memory overhead
def split_matrix_dims(dims, count):
    inner, outer = dims
    # vector case
    # if outer == 1:
    #     if inner % count == 0: # if we can evenly divide inner between all 'count' outputs
    #         return [(int(inner / count), 1)] * count
    #     else:
    #         extras = inner % count # we have to distribute these extras
    #         extra_element_distribution = []
    #         for i in range(count): # replace with list comprehension?
    #             if i < extras:
    #                 extra_element_distribution.append(1) # we want to distribute the extras evenly instead of tacking them on to to the last element
    #             else:
    #                 extra_element_distribution.append(0)
    #         return [tuple((int(inner / count)) + extra_element_distribution[i],1) for i in range(count)]
    # else: # matrix case
    # for now, we will only split into floor(sqrt(count)), which gives a square, to simplify splitting
    squares = int(math.pow(math.floor(math.sqrt(count)),2)) # find the largest number of squares we can make with an integer sqrt
    dim_divisor = int(round(math.sqrt(squares))) # we need to divide each dimension by this many to give our desired number of quadrants
    if inner % dim_divisor == 0 and outer % dim_divisor == 0:
        square_dims = [(int(inner / dim_divisor), int(outer / dim_divisor))] * squares
        idle = [(0,0)] * (count - squares)
        square_dims.extend(idle)
        return square_dims
    else:
        extras = inner % dim_divisor # we have to distribute these extras
        inner_extra_element_distribution = []
        for i in range(dim_divisor): # replace with list comprehension?
            if i < extras:
                inner_extra_element_distribution.append(1) # we want to distribute the extras evenly instead of tacking them on to to the last element
            else:
                inner_extra_element_distribution.append(0)
        extras = outer % dim_divisor # we have to distribute these extras
        outer_extra_element_distribution = []
        for i in range(dim_divisor): # replace with list comprehension?
            if i < extras:
                outer_extra_element_distribution.append(1) # we want to distribute the extras evenly instead of tacking them on to to the last element
            else:
                outer_extra_element_distribution.append(0)
        square_dims = [((int(inner / dim_divisor)) + inner_extra_element_distribution[j],(int(outer / dim_divisor)) + outer_extra_element_distribution[i]) for i in range(dim_divisor) for j in range(dim_divisor)] 
        idle = [(0,0)] * (count - squares)
        square_dims.extend(idle)
        return square_dims

# gets the two matrices that thread 'index' will 'multiply' together. The second matrix will be transposed. FIXME, this needs to give the right dimensions
def get_split_matrix(mat1, mat2_T, dims, index):
    start_index = 0
    inner_dim, outer_dim = dims[index]
    for dim in dims[:index]:
        inner, outer = dim
        start_index += inner
        if start_index % len(mat1[0]) == 0:
            start_index += len(mat1[0]) * (outer - 1)
    mat1_outer_start_index = math.floor(start_index / len(mat1[0]))
    mat2_outer_start_index = start_index % len(mat1[0])
    return_matrix1 = []
    return_matrix2 = []
    for i in range(mat1_outer_start_index, mat1_outer_start_index + outer_dim): # loop through outers
        return_matrix1.append(mat1[i]) # grab the whole inner
        
    for i in range(mat2_outer_start_index, mat2_outer_start_index + inner_dim):
        return_matrix2.append(mat2_T[i]) # mat2 is transposed, so we grab the inner as well

    return (return_matrix1, return_matrix2)

def reconstruct_split_matrices(matrices,original_dims):
    inner, outer = original_dims
    reconstructed_matrix = []
    reconstructed_matrix.extend(matrices[0]) # use the first matrix as our building block
    outer_offset = 0 # as we fill up blocks of inner matrices, we will need to index from a new 'zero'
    for matrix in matrices[1:]: # skip the first matrix because we included it above
        if not matrix:
            continue
        if outer_offset + 1 < len(reconstructed_matrix): # if we won't exceed the dimensions of our matrix by tacking on more to the existing inners
            for i in range(len(matrix)): # for each inner matrix
                reconstructed_matrix[outer_offset + i].extend(matrix[i]) # append it to the existing inner matrices
            if len(reconstructed_matrix[outer_offset]) == inner: # once we fill up the inners, we need to adjust the outer_offset
                outer_offset += len(matrix)
        else:
            reconstructed_matrix.extend(matrix)

    return reconstructed_matrix

def multiply_matrix_prep(mat1, mat2, process_count):
    dims = get_result_dims(mat1, mat2)

    split_dims = split_matrix_dims(dims,process_count)
    mat2_T = transpose(mat2)
    split_mats = []
    for i in range(len(split_dims)):
        split_mats.append(get_split_matrix(mat1,mat2_T,split_dims,i))

    return split_mats
    

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

    size_inner = int(argv[2])
    if len(argv) >= 4:
        size_outer = int(argv[3])
    else:
        size_outer = 1

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
    mat1 = generate_matrix(size_inner, size_outer)
    mat2 = generate_vector(size_inner, size_outer)

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