import math

import pytest
from qusim.theory.hilbert import Hilbert
from qusim.theory.matrix import Matrix


class TestHilbert:

    def test_init(self):
        val = 1 / math.sqrt(2)

        H1 = Hilbert(["+", "-"])
        H2 = Hilbert(2)
        H3 = Hilbert(basis={"0": [1, 0], "1": [0, 1]})
        H4 = Hilbert(["+", "-"], N=2)
        H5 = Hilbert(["alpha", "beta"], [[val, val], [val, -val]])
        H6 = Hilbert(["+", "-"], [[0, 1j], [1, 0]])

        assert H2 == H3

        assert str(H1) == "[|+>: [1.+0.j 0.+0.j],\n |->: [0.+0.j 1.+0.j]]"
        assert str(H2) == "[|0>: [1.+0.j 0.+0.j],\n |1>: [0.+0.j 1.+0.j]]"
        assert str(
            H4
        ) == "[|++>: [1.+0.j 0.+0.j 0.+0.j 0.+0.j],\n |+->: [0.+0.j 1.+0.j 0.+0.j 0.+0.j],\n |-+>: [0.+0.j 0.+0.j 1.+0.j 0.+0.j],\n |-->: [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]"
        assert str(
            H5
        ) == "[|alpha>: [0.70710678+0.j 0.70710678+0.j],\n |beta>: [ 0.70710678+0.j -0.70710678+0.j]]"
        assert repr(
            H6
        ) == "Hilbert[|+>: [0.+0.j 0.+1.j],\n        |->: [1.+0.j 0.+0.j]]"

        with pytest.raises(ValueError):
            Hilbert()

        with pytest.raises(ValueError):
            Hilbert(["alpha", "beta", "gamma"], [[val, val], [val, -val]])

        with pytest.raises(ValueError):
            Hilbert(["+", "-"], [[0, 1j], [val, val]])

    def test_kron(self):
        H1 = Hilbert(2)
        H2 = Hilbert(2, N=3)
        assert Hilbert.kron([H1, H1, H1]) == H2
        assert H1 * 3 == H2
        assert not H1 * H1 * H1 != H2

    def test_get_and_set(self):
        val = 1 / math.sqrt(2)
        H1 = Hilbert(["+", "-"])
        H2 = Hilbert(["+", "-"], [[val, val], [val, -val]])
        H3 = Hilbert(2)

        assert H1.names == ["+", "-"]
        assert H1.directions == Matrix.eye(2)
        print(repr(H1.basis["+"]))
        print(repr(Matrix([1, 0])))
        assert H1.basis["+"] == Matrix([1, 0])

        with pytest.raises(ValueError):
            H1.names = 3

        with pytest.raises(ValueError):
            H1.directions = Matrix.eye(3)

        with pytest.raises(ValueError):
            H1.directions = Matrix.ones(2)

        H1.names = 2
        assert H1 == H3

        H1.names = ["+", "-"]
        H1.directions = H2.directions
        assert H1 == H2

        H1.basis = {"0": [1, 0], "1": [0, 1]}
        assert H1 == H3
        H1.basis = {"+": [1, 0], "-": [0, 1]}
        assert H1 == H3

        with pytest.raises(ValueError):
            H1.basis = {"0": [1, 0, 0], "1": [0, 1, 0], "2": [0, 0, 1]}

        with pytest.raises(ValueError):
            H1.basis = {"+": [1, 0], "-": [1, 0]}
