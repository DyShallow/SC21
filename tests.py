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

if __name__ == '__main__':
    unittest.main()