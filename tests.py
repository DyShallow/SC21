import unittest
import parallel # careful, might be ambiguous
import multiprocessing # use multiprocessing
import random

def randcmp(_, x, y):
    return random.randrange(-1, 2)

class TestMatrixFunctions(unittest.TestCase):

    def test_transpose(self):
        matrix_size = 10
        mat1 = parallel.generate_matrix(matrix_size)
        mat1_T = parallel.transpose(mat1)
        mat1_T_np = parallel.transpose_numpy(mat1)
        mats_equal = parallel.matrices_equal(mat1_T, mat1_T_np)
        self.assertTrue(mats_equal)

    def test_matrix(self):
        generated_size = (10,10)
        split_count = 4
        mat = parallel.generate_matrix(generated_size[0], generated_size[1])
        mat2 = parallel.generate_matrix(generated_size[0], generated_size[1])
        dims2 = parallel.get_result_dims(mat, mat2, True)
        split_mat_dims = parallel.split_matrix_dims(dims2,split_count)
        self.assertEqual(split_mat_dims, [(5,5),(5,5),(5,5),(5,5)])

        split_mat_dims = parallel.split_matrix_dims(dims2,5)
        self.assertEqual(split_mat_dims, [(5,5),(5,5),(5,5),(5,5),(0,0)])

        split_mat_dims = parallel.split_matrix_dims(dims2,9)
        self.assertEqual(split_mat_dims, [(4, 4), (3, 4), (3, 4), (4, 3), (3, 3), (3, 3), (4, 3), (3, 3), (3, 3)])

        split_mat_dims = parallel.split_matrix_dims(dims2,10)
        self.assertEqual(split_mat_dims, [(4, 4), (3, 4), (3, 4), (4, 3), (3, 3), (3, 3), (4, 3), (3, 3), (3, 3), (0, 0)])

        # maybe replace this with parallel.matrix_multiply_prep
        mat2_T = parallel.transpose(mat2)
        mat2_T_numpy = parallel.transpose_numpy(mat2)
        self.assertTrue(parallel.matrices_equal(mat2_T, mat2_T_numpy))

        split_mats = []
        for i in range(len(split_mat_dims)):
            split_mats.append(parallel.get_split_matrix(mat,mat2_T,dims2,split_mat_dims,i,True))

        result_mats2 = []
        for mat_slice, mat2_T_slice in split_mats:
            result_mats2.append(parallel.multiply_matrix(mat_slice, mat2_T_slice))

        result2 = parallel.reconstruct_split_matrices(result_mats2,dims2)

        self.assertTrue(parallel.matrices_equal(result2, parallel.multiply_matrix_numpy(mat,mat2)))

    def generate_test_dims(self, count, rand_max):
        dims = []
        # some vectors
        vector_count = int(count / 10) + 1
        for _ in range(vector_count):
            dims.append((1, random.randint(1,rand_max), 1))
        
        # matrices
        for _ in range(count - vector_count):
            dims.append((random.randint(1,rand_max), random.randint(1,rand_max), random.randint(1,rand_max)))

        return dims
        
    def test_simulated_workflow(self):
        # setup
        
        number_of_processes = 10
        task_functions = [parallel.add_matrix, parallel.multiply_matrix]
        numpy_functions = [parallel.add_matrix_numpy, parallel.multiply_matrix_numpy]

        test_dims = self.generate_test_dims(10, 20) # generate 10 sets of dims, max dim of 20

        for ii in range(1,len(task_functions)): # TODO hard-code some special cases like 1-dimensional stuff (vector)
            for dims in test_dims:    
                function_pointer = task_functions[ii]
                
                result_outer_size, matching_size, result_inner_size = dims

                # generate items. For now we assume they are the same size TODO, support non-identical matrices
                mat1 = parallel.generate_matrix(matching_size, result_inner_size)
                
                if ii == 1: # need to transpose mat2 for multiplication
                    needs_transposition = True
                    mat2 = parallel.generate_matrix(result_outer_size, matching_size)
                else:
                    needs_transposition = False
                    mat2 = parallel.generate_matrix(matching_size, result_inner_size) # re-use mat1 dims in addition case

                numpy_result = numpy_functions[ii](mat1, mat2)

                result_dims = parallel.get_result_dims(mat1, mat2, needs_transposition)

                self.assertEqual(parallel.get_matrix_dims(numpy_result), result_dims)

                # # For more complex optimization, we might need to have some more synchronization
                manager = multiprocessing.Manager()
                shared_dict = manager.dict() # Processes will write their results to this list

                # Split up the work for each of the workers
                input_matrices = parallel.split_work(mat1, mat2, number_of_processes, transpose_result=needs_transposition)

                # start worker processes to perform the work
                workers = []
                for i in range(number_of_processes): # underscore indicates we do not care about the iterator variable itself
                    workers.append(multiprocessing.Process(target=parallel.execute_task, args=(i, function_pointer, shared_dict, input_matrices[i][0], input_matrices[i][1],))) # prepare workers
                for worker in workers:
                    worker.start() # start workers

                # join processes
                for worker in workers:
                    worker.join()
                #print('All processes joined!') # report all processes joined. If we don't get here, maybe one of the workers got stuck

                worker_results = []
                for i in range(number_of_processes):
                    if i in shared_dict:
                        worker_results.append(shared_dict[i])

                # recombine results
                result = parallel.reconstruct_split_matrices(worker_results, result_dims)

                self.assertTrue(parallel.get_matrix_dims(result), result_dims)

                #matching_elements = [[np == r for np,r in zip(np_arr, r_arr)] for np_arr, r_arr in zip(numpy_result, result)]

                self.assertTrue(parallel.matrices_equal(numpy_result, result))

        
if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = randcmp # run tests in random order
    unittest.main()