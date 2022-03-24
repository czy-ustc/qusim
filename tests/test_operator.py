import math
from qusim.theory.hilbert import Hilbert
from qusim.theory.matrix import Matrix
import pytest

from qusim.theory.state import Ket, Bra, State
from qusim.theory.operator import HamiltonOperator, MechanicalOperator, Operator, Sigma


class TestOperator:

    def test_init(self):
        op1 = Operator([[1, 0], [0, 0]])
        op2 = Operator({(0, 0): 1}, 2)
        op3 = Operator({((0, 0), 1)}, dim=2)

        assert op1 == op2 == op3

    def test_str(self):
        val = 1 / math.sqrt(2)

        op1 = Operator([[1, 0], [0, 1j]])
        op2 = Operator([[1, 0], [0, 1j]],
                       H=Hilbert(2, [[val, val], [val, -val]]))

        assert str(op1) == str(op2)
        assert repr(op1) == repr(op2)

    def test_matmul(self):
        op = Operator([[0, 1], [1, 0]])
        ket0 = Ket([1, 0])
        ket1 = Ket([[0], [1]])
        bra0 = Bra([[1], [0]])
        bra1 = Bra([0, 2j])

        assert op @ ket0 == ket1
        assert op @ ket1 == ket0
        assert bra0 @ op == bra1
        assert bra1 @ op == bra0
        assert op @ op == Matrix.eye(2)

    def test_commutator(self):
        sigmax, sigmay, sigmaz = Sigma.X(), Sigma.Y(), Sigma.Z()

        assert Operator.commutator(sigmax, sigmay) == sigmaz * 2j
        assert Operator.commutator(sigmay, sigmaz) == sigmax * 2j
        assert Operator.commutator(sigmaz, sigmax) == sigmay * 2j

    def test_kron(self):
        op1 = Operator(Matrix.eye(2))
        op2 = Operator(Matrix.eye(4))

        assert Operator.kron([op1, op1]) == op2

    def test_pow(self):
        I = Operator(Matrix.eye(2))
        sigmax, sigmay, sigmaz = Sigma.X(), Sigma.Y(), Sigma.Z()

        assert sigmax**2 == sigmay**2 == sigmaz**2 == I

    def test_check(self):
        sigmax, sigmay, sigmaz = Sigma.X(), Sigma.Y(), Sigma.Z()

        assert all(
            Operator.check_herm(sigma)
            and Operator.check_unit(sigma) == Operator.check_comp(sigma)
            for sigma in [sigmax, sigmay, sigmaz])

        op1 = Operator([[1, 1j], [1j, 1]])
        assert not Operator.check_herm(op1)
        assert not Operator.check_unit(op1)
        assert not Operator.check_comp(op1)

    def test_transform(self):
        op1 = Operator([[1, 2], [3, 4]])
        op2 = Operator([[4, 3], [2, 1]])

        assert op1.transform(Sigma.X()) == op2
        assert op2.transform(Sigma.X()) == op1

    def test_expm(self):
        op = Operator([[1.0, 2.0], [-1.0, 3.0]])
        assert op.cosm() + 1j * op.sinm() == (1j * op).expm()
        assert op.logm().expm() == op.expm().logm() == op

    def test_sqrtm(self):
        op = Operator([[1.0, 2.0], [-1.0, 3.0]])
        assert op.sqrtm()**2 == op

    def test_inv(self):
        op = Operator([[1.0, 2.0], [-1.0, 3.0]])
        assert op.inv.inv == op
        assert op.inv @ op == op @ op.inv == Sigma(0)

    def test_trace(self):
        op = Operator([[1j, 0], [0, 1]])
        assert op.trace == 1 + 1j

    def test_det(self):
        op = Operator([[1, 2j], [-3j, 4]])
        assert abs(op.det - (-2)) < 1e-12

    def test_norm(self):
        op = Operator([-4, -3, -2, -1, 0, 1, 2, 3, 4], dim=3)
        assert op.norm(1) == 7
        assert abs(op.norm(2) - 7.3484692283495345) < 1e-12
        assert op.norm("inf") == 9
