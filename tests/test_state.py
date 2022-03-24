import math

import pytest
from qusim.theory.hilbert import Hilbert
from qusim.theory.matrix import Matrix
from qusim.theory.qobj import Qobj
from qusim.theory.state import Bra, Ket, State


class TestState:

    def test_init(self):
        val = 1 / math.sqrt(2)

        s = State([1, -1])
        State([[1], [2]])
        State([1, 1], H=Hilbert(["+", "-"], [[val, val], [val, -val]]))

        with pytest.raises(ValueError):
            State([1, 0], H=Hilbert(3))
        with pytest.raises(ValueError):
            State([1, 0, 0, 0], (2, 2))

        assert s.type == "State"

    def test_str(self):
        val = 1 / math.sqrt(2)
        s1 = Ket([1, 1], H=Hilbert(2, [[val, val], [val, -val]]))
        s2 = Ket([1, 1])
        assert str(s1) == str(s2)
        s3 = Bra([1, 1j], H=Hilbert(["+", "-"], [[0, 1j], [1, 0]]))
        s4 = Bra([2, 2j], H=Hilbert(["+", "-"]))
        assert repr(s3) == repr(s4)

    def test_equal(self):
        val = 1 / math.sqrt(2)
        s1 = State([1, 1])
        s2 = State([1, 0], H=Hilbert(["+", "-"], [[val, val], [val, -val]]))
        assert (2 + 7.2j) * State([1, 0]) == State([2j, 0]) * 0.5j
        assert s1 == s2

        assert Ket([1, 1]) == Bra([1, 1]).ket
        assert Ket([1, 1]).bra == Bra([1, 1])
        assert Ket([1, 1, 0]) != Ket([1, 1])
        assert Ket([0, 1, 1]) != Ket([1, 1])
        assert Ket([1, 1j]) != Ket([1j, 1])

    def test_linear_operation(self):
        k1 = Ket([1, 0])
        k2 = Ket([0, 1])
        k3 = Ket([-1, -1])

        assert k1 + k2 == k3
        assert (k1 + k2) * 2 == k3
        assert (k1 - k2) * 2 != k3

    def test_mul(self):
        s1 = State([1, 1])
        s2 = State([[1], [1]])

        assert abs(s1 @ s2 - 1) < 1e-12
        assert s2 @ s1 == Qobj([[0.5, 0.5], [0.5, 0.5]])
        assert s1 @ Matrix.ones(2) == s1
        assert Matrix.ones(2) @ s2 == s2
        assert abs(s1 @ Matrix.ones(2) @ s2 - 1) < 1e-12

        with pytest.raises(ValueError):
            s1.bra @ s2.bra

        assert Bra([1, 0]) @ Ket([1, 0]) == 1

    def test_overlap(self):
        s1 = State([1, 1j])
        s2 = State([1, -1j])
        s3 = State([2 + 3j, -1 + 0.4j])
        s4 = State([-2, 3.7j])

        assert 1 - s1.overlap(s1) < 1e-12
        assert 1 - s2.overlap(s2) < 1e-12
        assert s1.overlap(s2) < 1e-12
        assert s2.overlap(s1) < 1e-12
        assert abs(s3.overlap(s4) - s4.overlap(s3)) < 1e-12

    def test_norm(self):
        s = State([1, 2, 3])
        assert abs(s.norm(1) - 1.6035674514745464) < 1e-12
        assert abs(s.norm(2) - 1) < 1e-12
        assert abs(s.norm("inf") - 0.8017837257372732) < 1e-12

    def test_kron(self):
        val = 1 / math.sqrt(2)
        s1 = Ket([1, 0], H=Hilbert(["+", "-"], [[val, val], [val, -val]]))
        s2 = Ket([1, 1])
        s3 = Ket([1, 1, 1, 1])
        Ket.kron([s1, s2]) == s3
        Bra.kron([s1.bra, s2.bra]) == s3.bra
