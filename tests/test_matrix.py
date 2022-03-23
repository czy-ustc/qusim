import numpy as np
import pytest
from qusim.theory.matrix import Matrix


class TestMatrix:

    def test_init(self):
        mat1 = Matrix({(0, 0): 1, (2, 2): 1}, shape=(3, 3))
        mat2 = Matrix([((0, 0), 1), ((2, 2), 1)], shape=(3, 3))
        mat3 = Matrix([[1, 0, 0], [0, 0, 0], [0, 0, 1]])

        assert mat1 == mat3
        assert not mat2 != mat3
        assert Matrix.eye(2) == Matrix([[1, 0], [0, 1]])
        assert Matrix.ones(2) == Matrix([[1, 1], [1, 1]])
        assert Matrix.zeros(2) == Matrix([[0, 0], [0, 0]])
        assert Matrix.ones((1, 3)) == Matrix([1, 1, 1])
        assert Matrix.zeros((2, 1)) == Matrix([0, 0], (2, 1))

        with pytest.raises(ValueError):
            Matrix([1, 2, 3, 4, 5, 6], shape=(2, 3))
        with pytest.raises(ValueError):
            Matrix.ones((2, 4))
        with pytest.raises(ValueError):
            Matrix.zeros((2, 4))

        assert Matrix([1]) == 1
        assert Matrix([[1j]]) == 1j
        assert Matrix({(0, 0): 1 + 1j}, (1, 1)) == 1 + 1j
        assert Matrix([((0, 0), 1 + 1j)], (1, 1)) == 1 + 1j

    def test_str(self):
        mat = Matrix([1, 2])
        mat_str = "[[1.+0.j 2.+0.j]]"
        mat_repr = "Matrix([[1.+0.j, 2.+0.j]])"

        assert str(mat) == mat_str
        assert repr(mat) == mat_repr

    def test_equal(self):
        mat1 = Matrix([1, 1])
        mat2 = Matrix([1, 1], shape=(2, 1))
        assert mat1 != mat2
        assert mat1 == mat2.H

    def test_linear_operation(self):
        mat1 = Matrix([1, 2])
        mat2 = Matrix([1, 2], shape=(2, 1))

        assert mat1 == mat2.T
        assert mat1 * 2 - mat2.T + mat1 == 2 * mat1

        with pytest.raises(ValueError):
            mat1 + mat2
        with pytest.raises(ValueError):
            mat1 - mat2

    def test_matmul(self):
        mat1 = Matrix([1, 2])
        mat2 = Matrix([1, 2], shape=(2, 1))
        mat3 = Matrix([[1, 2], [2, 4]])

        assert mat1 @ mat2 == 5
        assert mat2 @ mat1 == mat3

    def test_kron(self):
        mat1 = Matrix([1, 2])
        mat2 = Matrix([1, 2, 2, 4])

        assert mat1.kron(mat1) == mat2

    def test_T(self):
        mat1 = Matrix([[1, 2, 3], [4, 5, 6]])
        mat2 = Matrix([[1, 4], [2, 5], [3, 6]])
        assert mat1.T == mat2

    def test_conj(self):
        mat1 = Matrix([1 + 1j, 2 + 2j])
        mat2 = Matrix([1 - 1j, 2 - 2j])
        assert mat1.conj == mat2

    def test_H(self):
        mat1 = Matrix([1j, 2j])
        mat2 = Matrix([[-1j], [-2j]])
        mat3 = Matrix([[1, 2], [3j, 4j]])
        mat4 = Matrix([[1, -3j], [2, -4j]])

        assert mat1.H == mat2
        assert mat2.H == mat1
        assert mat3.H == mat4

    def test_resize(self):
        mat1 = Matrix([[1, 2, 3], [4, 5, 6]])
        mat2 = Matrix([[1, 2], [3, 4], [5, 6]])
        mat3 = Matrix([1, 2, 3, 4, 5, 6], (6, 1))

        assert mat1.reshape((3, 2)) == mat2
        mat1.resize((6, 1))
        assert mat1 == mat3

    def test_nonzero(self):
        mat1 = Matrix({(0, 0): 1, (2, 2): 1}, shape=(3, 3))
        mat2 = Matrix([((0, 0), 1), ((2, 2), 1)], shape=(3, 3))
        mat3 = Matrix(mat1.nonzero(), shape=(3, 3))

        assert mat2 == mat3

    def test_get_set(self):
        mat = Matrix([[1, 2], [3, 4]])
        mat[(0, 0)] = -1
        array = np.array([[-1, 2], [3, 4]], dtype=complex)

        assert (mat.toarray() == array).all()
        assert mat.tobytes() == array.tobytes()
        assert mat.size == 4
        assert mat.tolist() == [[-1 + 0j, 2 + 0j], [3 + 0j, 4 + 0j]]

        with pytest.raises(ValueError):
            mat.array = np.array([1, 2, 3])
        mat.array = Matrix.ones(2)
        assert mat == Matrix.ones(2)
