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

    def test_split_and_reconstruct(self):
        matrix_size = 10
        split_count = 2
        vec = parallel.generate_vector(matrix_size)
        vec2 = parallel.generate_vector(matrix_size)
        split_vec_dims = parallel.split_matrix_dims(parallel.get_dims(vec),split_count)
        self.assertEqual(split_vec_dims,((5,1),(5,1)))

        split_mats = []
        for i in range(len(split_vec_dims)):
            split_mats.append(parallel.get_split_matrix(vec,vec2,split_vec_dims,i))

        split_mat1 = []
        for tup in split_mats:
            first, second = tup
            split_mat1.append(first)

        reconstruction = parallel.reconstruct_split_matrices(split_mat1,(10,1))
        self.assertTrue(parallel.vectors_equal(vec, reconstruction))



if __name__ == '__main__':
    unittest.main()