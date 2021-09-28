import unittest
import parallel # careful, might be ambiguous

class TestMatrixFunctions(unittest.TestCase):

    def test_transpose(self):
        matrix_size = 10
        mat1 = parallel.generate_matrix(matrix_size)
        mat1_T = parallel.transpose(mat1)
        mat1_T_np = parallel.transpose_numpy(mat1)
        mats_equal = parallel.matrices_equal(mat1_T, mat1_T_np)
        self.assertTrue(mats_equal)

    def test_split_and_reconstruct_vector(self):
        matrix_size = 10
        split_count = 2
        vec = parallel.generate_vector(matrix_size)
        vec2 = parallel.generate_vector(matrix_size)
        split_vec_dims = parallel.split_matrix_dims(parallel.get_dims(vec),split_count)
        self.assertEqual(split_vec_dims,[(5,1),(5,1)])

        split_mats = []
        for i in range(len(split_vec_dims)):
            split_mats.append(parallel.get_split_matrix(vec,vec2,split_vec_dims,i))

        split_mat1 = []
        for tup in split_mats:
            first, second = tup
            split_mat1.append(first)

        reconstruction = parallel.reconstruct_split_matrices(split_mat1,(10,1))
        self.assertTrue(parallel.vectors_equal(vec, reconstruction))

    def test_split_and_reconstruct_matrix(self):
        dims = (10,10)
        split_count = 4
        split_mat_dims = parallel.split_matrix_dims(dims,split_count)
        self.assertEqual(split_mat_dims, [(5,5),(5,5),(5,5),(5,5)])

        split_mat_dims = parallel.split_matrix_dims(dims,5)
        self.assertEqual(split_mat_dims, [(5,5),(5,5),(5,5),(5,5),(0,0)])

        split_mat_dims = parallel.split_matrix_dims(dims,9)
        self.assertEqual(split_mat_dims, [(4, 4), (3, 4), (3, 4), (4, 3), (3, 3), (3, 3), (4, 3), (3, 3), (3, 3)])

        split_mat_dims = parallel.split_matrix_dims(dims,10)
        self.assertEqual(split_mat_dims, [(4, 4), (3, 4), (3, 4), (4, 3), (3, 3), (3, 3), (4, 3), (3, 3), (3, 3), (0, 0)])

        mat = parallel.generate_matrix(dims[0], dims[1])
        mat2 = parallel.generate_matrix(dims[0], dims[1])

        # maybe replace this with parallel.matrix_multiply_prep
        mat2_T = parallel.transpose(mat2)
        mat2_T_numpy = parallel.transpose_numpy(mat2)
        self.assertTrue(parallel.matrices_equal(mat2_T, mat2_T_numpy))

        split_mats = []
        for i in range(len(split_mat_dims)):
            split_mats.append(parallel.get_split_matrix(mat,mat2_T,split_mat_dims,i))

        result_mats = []
        for mat, mat2_T in split_mats:
            result_mats.append(parallel.multiply_matrix_elementwise(mat, mat2_T))

        result = parallel.reconstruct_split_matrices(result_mats,dims)

        self.assertTrue(parallel.matrices_equal(result, parallel.multiply_matrix_numpy(mat,mat2)))
        
if __name__ == '__main__':
    unittest.main()