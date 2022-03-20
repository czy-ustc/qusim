# Author       : czy
# Date         : 2022-03-17 23:21:25
# LastEditTime : 2022-03-17 23:21:25
# FilePath     : /qusim/core/base/operator.py
# Description  : The operator acting on the state vector.
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>

from ctypes import Union
from typing import Dict, Optional, Tuple

from qusim.theory.qobj import Qobj
from qusim.theory.state import State
from qusim.theory.type import MatData


class Operator(Qobj):
    """
    Linear operator in Hilbert space.

    The basis of matrix mechanics 
    corresponds to the linear mapping in Hilbert space.

    """

    __type__ = "Operator"

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(self, data: MatData, dim: Optional[int] = None) -> None:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Multiplication Operations

    @classmethod
    def commutator(cls, src1: "Operator", src2: "Operator") -> "Operator":
        """Commutator of operator."""
        raise NotImplementedError

    def __pow__(self, other: int) -> "Operator":
        """Power: Only a square matrix can be exponentiated."""
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Check

    @classmethod
    def check_herm(cls, oper: "Operator") -> bool:
        """Check if operator is Hermitian."""
        raise NotImplementedError

    @classmethod
    def check_unit(cls, oper: "Operator") -> bool:
        """Check if operator is Unitary."""
        raise NotImplementedError

    @classmethod
    def check_comp(cls, oper: "Operator") -> bool:
        """
        Check whether the operator meets the completeness condition, 
        that is, whether the operator is a mechanical operator.
        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Matrix Transformation

    def transform(self, matrix: MatData) -> "Operator":
        """
        A basis transformation defined by matrix.

        Notes
        -----
        `matrix` must be a unitary matrix.
        
        """
        raise NotImplementedError

    def sinm(self) -> "Operator":
        """Sine of operator."""
        raise NotImplementedError

    def cosm(self) -> "Operator":
        """Cosine of operator."""
        raise NotImplementedError

    def expm(self) -> "Operator":
        """Matrix exponential of operator."""
        raise NotImplementedError

    def logm(self) -> "Operator":
        """Matrix logarithm of operator."""
        raise NotImplementedError

    def sqrtm(self) -> "Operator":
        """Matrix sqrt of operator."""
        raise NotImplementedError

    @property
    def inv(self) -> "Operator":
        """Inverse of matrix."""
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Metric

    @property
    def trace(self) -> complex:
        """Trace of matrix."""
        raise NotImplementedError

    @property
    def det(self) -> float:
        """Determinant of matrix."""
        raise NotImplementedError

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
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Other Methods

    @property
    def eig(self) -> Dict[complex, "State"]:
        """Eigenvalues and eigenvectors of matrix."""
        raise NotImplementedError


class HermiteOperator(Operator):
    """Hermite Operator."""

    __type__ = "Hermite Operator"

    def matrix_element(self, a: State, b: State) -> complex:
        """Matrix element `<a|A|b>`."""
        raise NotImplementedError

    @property
    def H(self) -> "HermiteOperator":
        """Return `self`."""
        raise NotImplementedError

    @property
    def eig(self) -> Dict[float, "State"]:
        """
        Eigenvalues and eigenvectors of matrix (Eigenvalues are real numbers).
        """
        raise NotImplementedError


class UnitaryOperator(HermiteOperator):
    """Unitary Operator."""

    __type__ = "Unitary Operator"

    @property
    def inv(self) -> "Operator":
        """Inverse of matrix (equivalent to adjoint of matrix)."""
        raise NotImplementedError


class MechanicalOperator(HermiteOperator):
    """Mechanical Operator."""

    __type__ = "Mechanical Operator"

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
        raise NotImplementedError

    def uncertainty(self, state: "State") -> float:
        """
        Uncertainty of mechanical quantity ensemble, 
        that is `sqrt(<A^2>-<A>^2)`.

        Parameters
        ----------
        state : State
            Quantum state of the system.

        Returns
        -------
        result : float
            Uncertainty of mechanical quantity.

        """
        raise NotImplementedError


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
        raise NotImplementedError


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

    """
    __type__ = "Pauli Operator"

    def __init__(self, type: Union[int, str, Tuple[float, float,
                                                   float]]) -> None:
        raise NotImplementedError

    @classmethod
    def X(cls) -> "Sigma":
        """Return sigmax."""
        raise NotImplementedError

    @classmethod
    def Y(cls) -> "Sigma":
        """Return sigmay."""
        raise NotImplementedError

    @classmethod
    def Z(cls) -> "Sigma":
        """Return sigmaz."""
        raise NotImplementedError


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
    
    """

    __type__ = "Projection Operator"

    def __init__(self, index: int, dim: int) -> None:
        raise NotImplementedError
