# Author: Dylan Shallow
# SC21 student outreach workshop
# This program is intended to demonstrate the performance potential of parallelism for certain tasks

import multiprocessing
from multiprocessing.managers import SyncManager
import sys      # accept arguments
import json     # read and write JSON
from collections import defaultdict # map one key to multiple values in a dict-like object
from multiprocessing import Process, JoinableQueue, Lock, Manager # use multiprocessing to improve throughput. JoinableQueue and Lock for synchronization. Manager for shared objects, in this case a dict to keep track of indices
import time # record execution time and provide estimates

# globals. We group these variables here for easy editing.
# For variables that will only change rarely, it can be easier to place them here than enter them in as arguments every time we run the program.
performance_log_file = 'performance_log.txt'
number_of_processes = 4 # TODO(Dylan): accept as argument

# simple usage output. TODO(Dylan): more descriptive, e.g. list and describe each argument
def usage():
    print(r'Usage: \npy .\python-filename.py [args]')

def add_vector(vec1, vec2): # TODO(Dylan): Maybe make these matrices and check dimensionality so that we avoid confusion of passing 'mat' to 'vec' in the worker prepare step
    for (i,j) in zip
    pass

def multiply_vector(vec1, vec2):
    pass

def add_matrix(mat1, mat2):
    pass

def multiply_matrix(mat1, mat2):
    pass

def transpose(mat):
    pass

def main(argv):
    # parse arguments. If we aren't able to parse them properly, remind the user of proper usage and quit
    if len(argv) not in [2]: # check length. We use 'in' syntax in case we want to add optional arguments, in which case the length of the argument list might have more than one valid value
        usage()
        quit()

    # Use function pointers so that we can re-use the code which handles parallelism
    tasks = ['add_vector', 'multiply_vector', 'add_matrix', 'multiply_matrix'] # supported operations
    task_functions = [add_vector, multiply_vector, add_matrix, multiply_matrix] # functions corresponding to tasks
    if(argv[1] in tasks):
        function_pointer = task_functions[tasks.index(argv[1])]
    else:
        print(f'Invalid task specified! {argv[1]}')
        usage()
        quit()

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