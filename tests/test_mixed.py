from math import sqrt
from qusim.theory.operator import Sigma
import pytest
from qusim.theory.hilbert import Hilbert
from qusim.theory.mixed import DensityMatrix, SuperOperator
from qusim.theory.state import Ket


class TestMixed:

    def test_init(self):
        s0 = Ket([1, 0])
        s1 = Ket([0, 1])
        dm1 = DensityMatrix(Ket.kron([s0, s1]))
        dm2 = DensityMatrix.kron([DensityMatrix(s0), DensityMatrix(s1)])
        dm3 = DensityMatrix(s0, (s1, 1))
        dm4 = DensityMatrix(data=[[0.5, 0], [0, 0.5]])
        assert dm1 == dm2
        assert repr(dm3) == repr(dm4)

        p = 0.05
        k1 = SuperOperator([[sqrt(1 - p), 0], [0, sqrt(1 - p)]], [[0, sqrt(p)], [sqrt(p), 0]])
        k2 = SuperOperator((Sigma(0), 1 - p), (Sigma(1), p))
        assert k1 == k2

    def test_linear_operation(self):
        dm = DensityMatrix(Ket([1, 1]))

        with pytest.raises(TypeError):
            dm + dm
        with pytest.raises(TypeError):
            dm - dm
        with pytest.raises(TypeError):
            dm * 2
        with pytest.raises(TypeError):
            1j * dm

    def test_overlap(self):
        dm1 = DensityMatrix(Ket([1, 0]))
        dm2 = DensityMatrix(Ket([0, 1]))

        assert dm1.overlap(dm2) == 0
        assert dm1.overlap(dm1) == dm2.overlap(dm2) == 1

        dm3 = DensityMatrix(Ket([1, 0]), Ket([0, 1]))
        assert dm3.overlap(dm1) == dm3.overlap(dm2) == dm3.overlap(dm3) == 0.5

        dm4 = DensityMatrix((Ket([1, 0]), 0.75), (Ket([0, 1]), 0.25))
        assert dm4.overlap(dm1) == 0.75
        assert dm4.overlap(dm2) == 0.25
        assert dm4.overlap(dm3) == 0.5
        assert dm4.overlap(dm4) == 0.625

    def test_metric(self):
        dm1 = DensityMatrix(Ket([1, 0]))
        dm2 = DensityMatrix(Ket([0, 1]))
        dm3 = DensityMatrix(Ket([1, 0]), Ket([0, 1]))
        assert dm1.trace == dm2.trace == dm3.trace == 1
        assert dm3.purity < dm1.purity == dm2.purity == 1
        assert dm1.det == dm2.det == 0
        assert dm3.det == 0.25

    def test_dm2ket(self):
        s = Ket([1 + 1j, 1 - 1j])
        dm = DensityMatrix(s)
        assert dm.ket == s.ket
        assert dm.bra == s.bra

    def test_ptrace(self):
        s0 = Ket([1, 0])
        s1 = Ket([0, 1])
        s2 = Ket([1, 1j], H=Hilbert(2, [[1, 0], [0, 1j]]))
        dm0 = DensityMatrix(s0)
        dm1 = DensityMatrix(s1)
        dm2 = DensityMatrix(s2)
        dm01 = DensityMatrix.kron([dm0, dm1])
        dm02 = DensityMatrix.kron([dm0, dm2])
        dm12 = DensityMatrix.kron([dm1, dm2])
        dm012 = DensityMatrix.kron([dm0, dm1, dm2])
        assert dm012.ptrace(0) == dm0
        assert dm012.ptrace(1) == dm1
        assert dm012.ptrace(2) == dm2
        assert dm012.ptrace([0, 1]) == dm01
        assert dm012.ptrace([0, 2]) == dm02
        assert dm012.ptrace([1, 2]) == dm12
