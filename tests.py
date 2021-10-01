import unittest
import parallel # careful, might be ambiguous
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

    def test_vector(self):
        matrix_size = 10
        split_count = 2
        vec = parallel.generate_vector(matrix_size)
        vec2 = parallel.transpose(parallel.generate_vector(matrix_size))
        split_vec_dims = parallel.split_matrix_dims(parallel.get_result_dims(vec, vec2),split_count)
        self.assertEqual(split_vec_dims,[(1,1),(0,0)])

        vec2_T = parallel.transpose(vec2)
        split_mats = []
        for i in range(len(split_vec_dims)):
            split_mats.append(parallel.get_split_matrix(vec,vec2_T,split_vec_dims,i))

        result_mats = []
        for mat, mat2_T in split_mats:
            result_mats.append(parallel.multiply_matrix(mat, mat2_T))

        result = parallel.reconstruct_split_matrices(result_mats,split_vec_dims)

        self.assertTrue(parallel.matrices_equal(result, parallel.multiply_matrix_numpy(vec,vec2)))

    def test_matrix(self):
        generated_size = (10,10)
        split_count = 4
        mat = parallel.generate_matrix(generated_size[0], generated_size[1])
        mat2 = parallel.generate_matrix(generated_size[0], generated_size[1])
        dims2 = parallel.get_result_dims(mat, mat2)
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
            split_mats.append(parallel.get_split_matrix(mat,mat2_T,split_mat_dims,i))

        result_mats2 = []
        for mat_slice, mat2_T_slice in split_mats:
            result_mats2.append(parallel.multiply_matrix(mat_slice, mat2_T_slice))

        result2 = parallel.reconstruct_split_matrices(result_mats2,dims2)

        self.assertTrue(parallel.matrices_equal(result2, parallel.multiply_matrix_numpy(mat,mat2)))
        
if __name__ == '__main__':
    unittest.TestLoader.sortTestMethodsUsing = randcmp # run tests in random order
    unittest.main()