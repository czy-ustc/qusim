# Author       : czy
# Date         : 2022-03-17 23:21:25
# LastEditTime : 2022-03-17 23:21:25
# FilePath     : /qusim/core/base/operator.py
# Description  : The operator acting on the state vector.
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>

import math
from functools import cached_property, reduce
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from qusim.theory.hilbert import Hilbert
from qusim.theory.matrix import Matrix
from qusim.theory.qobj import Qobj
from qusim.theory.state import State
from qusim.theory.type import MatData, Shape, __eps__
from scipy.linalg import cosm, expm, logm, sinm, sqrtm


class Operator(Qobj):
    """
    Linear operator in Hilbert space.

    The basis of matrix mechanics 
    corresponds to the linear mapping in Hilbert space.

    """

    __type__ = "Operator"

    # ----------------------------------------------------------------------
    # Constructors

    def __new__(cls,
                data: MatData,
                shape: Optional[Shape] = None,
                *args,
                **kwargs) -> Any:
        if isinstance(shape, int):
            shape = (shape, shape)
        elif "dim" in kwargs:
            shape = (kwargs["dim"], kwargs["dim"])
        return super().__new__(cls, data, shape=shape, *args, **kwargs)

    def __init__(
        self,
        data: MatData,
        dim: Optional[int] = None,
        H: Optional[Hilbert] = None,
    ) -> None:
        if dim is not None:
            super().__init__(data, shape=(dim, dim), H=H)
        else:
            super().__init__(data, H=H)

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:
        with np.printoptions(precision=2):
            basis = self.space.directions
            data = basis.array.conj() @ self.array @ basis.array.T
            return np.round(data, 2).__str__()

    def __repr__(self) -> str:
        string = self.__str__()
        return f"<[{self.type}]>\n{string}"

    # ----------------------------------------------------------------------
    # Basic Operations In Linear Space

    def __add__(self, other: "Operator") -> "Operator":
        if (self.shape == other.shape):
            return Operator(self.array.__add__(other.array))
        else:
            raise ValueError(
                "add: only matrices with the same shape can be added")

    def __sub__(self, other: "Operator") -> "Operator":
        if (self.shape == other.shape):
            return Operator(self.array.__sub__(other.array))
        else:
            raise ValueError(
                "sub: only matrices with the same shape can be subtracted")

    def __mul__(self, other: complex) -> "Operator":
        return Operator(self.array.__mul__(other))

    def __rmul__(self, other: complex) -> "Operator":
        return Operator(self.array.__mul__(other))

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def __matmul__(
            self, other: Union["Operator", State]) -> Union["Operator", State]:
        if isinstance(other, State):
            return other.__class__(self.array @ other.array, H=self.space)
        else:
            return self.__class__(self.array @ other.array, H=self.space)

    @classmethod
    def commutator(cls, src1: "Operator", src2: "Operator") -> "Operator":
        """
        Commutator of operator.
        
        Notes
        -----
        If the Hilbert space of the two operators is different, 
        the default Hilbert space is used.
        
        """
        if src1.shape != src2.shape:
            raise ValueError("dimension mismatch")

        array = src1.array @ src2.array - src2.array @ src1.array
        if src1.space == src2.space:
            return Operator(array, H=src1.space)
        else:
            return Operator(array)

    @classmethod
    def kron(cls, opers: List["Operator"]) -> "Operator":
        """Direct product."""
        H = Hilbert.kron([op.space for op in opers])
        data = reduce(lambda x, y: np.kron(x, y), [op.array for op in opers])
        return Operator(data, H=H)

    def __pow__(self, other: int) -> "Operator":
        """Power: Only a square matrix can be exponentiated."""
        return Operator(np.linalg.matrix_power(self.array, other),
                        H=self.space)

    # ----------------------------------------------------------------------
    # Check

    @classmethod
    def check_herm(cls, oper: "Operator") -> bool:
        """Check if operator is Hermitian."""
        return np.allclose(oper.array, np.conj(oper.array).T, atol=__eps__)

    @classmethod
    def check_unit(cls, oper: "Operator") -> bool:
        """Check if operator is Unitary."""
        return np.allclose(oper.array @ np.conj(oper.array).T,
                           np.eye(oper.dim),
                           atol=__eps__)

    @classmethod
    def check_comp(cls, oper: "Operator") -> bool:
        """
        Check whether the operator meets the completeness condition, 
        that is, whether the operator is a mechanical operator.
        """
        if cls.check_herm(oper):
            eigenvectors = np.linalg.eig(oper.array)[1]
            return np.allclose(eigenvectors @ np.conj(eigenvectors).T,
                               np.eye(oper.dim),
                               atol=__eps__)
        return False

    # ----------------------------------------------------------------------
    # Matrix Transformation

    def transform(self,
                  matrix: MatData,
                  shape: Optional[Shape] = None) -> "Operator":
        """
        A basis transformation defined by matrix.

        Notes
        -----
        `matrix` must be a unitary matrix.
        
        """
        mat = Matrix(matrix, shape)
        if (mat.H @ mat) != Matrix.eye(self.dim):
            raise ValueError("'matrix' must be a unitary matrix")

        if mat.shape != self.shape:
            raise ValueError("dimension mismatch")

        return self.__class__(mat.H.array @ self.array @ mat.array,
                              H=self.space)

    def sinm(self) -> "Operator":
        """Sine of operator."""
        return self.__class__(sinm(self.array), H=self.space)

    def cosm(self) -> "Operator":
        """Cosine of operator."""
        return self.__class__(cosm(self.array), H=self.space)

    def expm(self) -> "Operator":
        """Matrix exponential of operator."""
        return self.__class__(expm(self.array), H=self.space)

    def logm(self) -> "Operator":
        """Matrix logarithm of operator."""
        return self.__class__(logm(self.array), H=self.space)

    def sqrtm(self) -> "Operator":
        """Matrix sqrt of operator."""
        return self.__class__(sqrtm(self.array), H=self.space)

    @cached_property
    def inv(self) -> "Operator":
        """Inverse of matrix."""
        return self.__class__(np.linalg.inv(self.array), H=self.space)

    # ----------------------------------------------------------------------
    # Metric

    @property
    def trace(self) -> complex:
        """Trace of matrix."""
        return self.array.trace()

    @property
    def det(self) -> float:
        """Determinant of matrix."""
        return np.linalg.det(self.array)

    def norm(self, ord: Union[int, str] = 2) -> float:
        """
        Return norm for operator.

        Parameters
        ----------
        ord : int or str, default 2
            Order of the norm.

        Returns
        -------
        result : float
            Norm of the operator matrix.

        """
        if isinstance(ord, int):
            return np.linalg.norm(self.array, ord=ord, axis=None)
        elif ord == "inf":
            return np.linalg.norm(self.array, ord=np.inf, axis=None)
        else:
            return np.linalg.norm(self.array, ord=int(ord), axis=None)

    # ----------------------------------------------------------------------
    # Container Methods

    def __setitem__(self, key: Any, value: Any) -> None:
        """The operator matrix element cannot be modified."""
        return NotImplemented

    # ----------------------------------------------------------------------
    # Other Methods

    @property
    def eig(self) -> Dict[complex, "State"]:
        """Eigenvalues and eigenvectors of matrix."""
        eigenvalues, eigenvectors = np.linalg.eig(self.array)
        return {k: v for k, v in zip(eigenvalues, eigenvectors)}


class HermiteOperator(Operator):
    """Hermite Operator."""

    __type__ = "Hermite Operator"

    def __init__(self,
                 data: MatData,
                 dim: Optional[int] = None,
                 H: Optional[Hilbert] = None,
                 check: bool = True) -> None:
        super().__init__(data, dim=dim, H=H)

        if check and not self.check_herm(self):
            raise ValueError("not Hermite operator")

    def __matmul__(self, other: Union[Operator,
                                      State]) -> Union[Operator, State]:
        if isinstance(other, HermiteOperator):
            return self.__class__(self.array @ other.array,
                                  H=self.space,
                                  check=False)
        else:
            return super().__matmul__(other)

    @property
    def H(self) -> "HermiteOperator":
        """Return `self`."""
        return self

    @property
    def eig(self) -> Dict[float, "State"]:
        """
        Eigenvalues and eigenvectors of matrix (Eigenvalues are real numbers).
        """
        return super().eig()


class UnitaryOperator(HermiteOperator):
    """Unitary Operator."""

    __type__ = "Unitary Operator"

    def __init__(self,
                 data: MatData,
                 dim: Optional[int] = None,
                 H: Optional[Hilbert] = None,
                 check: bool = True) -> None:
        super().__init__(data, dim=dim, H=H)

        if check and not self.check_unit(self):
            raise ValueError("not unitary operator")

    def __matmul__(self, other: Union[Operator,
                                      State]) -> Union[Operator, State]:
        if isinstance(other, UnitaryOperator):
            return self.__class__(self.array @ other.array,
                                  H=self.space,
                                  check=False)
        else:
            return super().__matmul__(other)

    @property
    def inv(self) -> "Operator":
        """Inverse of matrix (equivalent to adjoint of matrix)."""
        return self.H


class MechanicalOperator(HermiteOperator):
    """Mechanical Operator."""

    __type__ = "Mechanical Operator"

    def __init__(self,
                 data: MatData,
                 dim: Optional[int] = None,
                 H: Optional[Hilbert] = None,
                 check=True) -> None:
        super().__init__(data, dim=dim, H=H)

        if check and not self.check_comp(self):
            raise ValueError("not mechanical operator")

    def __matmul__(self, other: Union[Operator,
                                      State]) -> Union[Operator, State]:
        if isinstance(other, MechanicalOperator):
            return self.__class__(self.array @ other.array,
                                  H=self.space,
                                  check=False)
        else:
            return super().__matmul__(other)

    def expect(self, state: "State") -> float:
        """
        Average value of mechanical quantity ensemble, 
        that is `<psi|A|psi>`.

        Parameters
        ----------
        state : State
            Quantum state of the system.

        Returns
        -------
        result : float
            Expected value of mechanical quantity.

        """
        return state.bra.array @ self.array @ state.ket.array


class HamiltonOperator(MechanicalOperator):
    """Hamilton Operator."""

    __type__ = "Hamilton Operator"

    def evolve(self, state: State, time: Optional[float] = None) -> State:
        """
        The state vector of the system 
        follows the Schrodinger equation and evolves with time.

        Parameters
        ----------
        state : State
            Quantum state of the system.
        time : float, optional
            Evolution time. If time is not `None`, 
            a time evolution operator will be generated 
            according to the Hamiltonian and evolution time, 
            otherwise it means that it itself is a time evolution operator.

        Returns
        -------
        result : State
            New state vector.

        """
        if time is not None:
            U = HamiltonOperator(self.array * (1j * time),
                                 H=self.space,
                                 check=False).expm()
        else:
            U = self
        return U @ state


class Sigma(UnitaryOperator, MechanicalOperator):
    """
    Pauli operator.

    A set of matrices including sigmax, sigmay and sigmaz.

    Parameters
    ----------
    type : int, str of tuple
        If `type` is str, `I`, `X`, `Y`, `Z` 
        represent I, sigmax, sigmay and sigmaz respectively; 
        else if `type` is int, 0, 1, 2, 3 
        represent I, sigmax, sigmay and sigmaz respectively; 
        otherwise, `type` represents a vector in three-dimensional space 
        and is recorded as `[a, b. c]`, thus generate matrix A 
        (a * sigmax + b * sigmay + c * sigmaz).
    H : Hilbert, optional
        Hilbert space.

    """
    __type__ = "Pauli Operator"

    def __init__(self,
                 type: Union[int, str, Tuple[float, float, float]],
                 H: Optional[Hilbert] = None) -> None:
        mapping = {"I": 0, "X": 1, "Y": 2, "Z": 3}
        if isinstance(type, str):
            type = mapping[type]

        if isinstance(type, int):
            if type == 0:
                super().__init__([[1, 0], [0, 1]], H=H, check=False)
            elif type == 1:
                super().__init__([[0, 1], [1, 0]], H=H, check=False)
            elif type == 2:
                super().__init__([[0, -1j], [1j, 0]], H=H, check=False)
            elif type == 3:
                super().__init__([[1, 0], [0, -1]], H=H, check=False)
        else:
            array = np.array(type)
            array = array * (1 / np.linalg.norm(array, 2))
            sigma = np.array([[[0, 1], [1, 0]], [[0, -1j], [1j, 0]],
                              [[1, 0], [0, -1]]])
            data = np.sum(array.reshape((3, 1, 1)) * sigma, axis=0)
            super().__init__(data, H=H, check=False)

    @classmethod
    def X(cls, H: Optional[Hilbert] = None) -> "Sigma":
        """Return sigmax."""
        return Sigma(1, H=H)

    @classmethod
    def Y(cls, H: Optional[Hilbert] = None) -> "Sigma":
        """Return sigmay."""
        return Sigma(2, H=H)

    @classmethod
    def Z(cls, H: Optional[Hilbert] = None) -> "Sigma":
        """Return sigmaz."""
        return Sigma(3, H=H)


class ProjectionOperator(MechanicalOperator):
    """
    Projection Operator.

    Project the state vector onto one of the eigenstates.

    Parameters
    ----------
    index : int
        Project the state vector onto the `index` eigenstate.
    dim : int
        Dimension of Hilbert space.
    H : Hilbert, optional
        Hilbert space.
    
    """

    __type__ = "Projection Operator"

    def __init__(self,
                 index: int,
                 dim: int,
                 H: Optional[Hilbert] = None) -> None:
        super().__init__({(index, index): 1}, dim=dim, H=H)
