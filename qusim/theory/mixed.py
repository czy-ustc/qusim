# Author       : czy
# Date         : 2022-03-18 20:23:22
# LastEditTime : 2022-03-18 20:28:08
# FilePath     : /qusim/theory/mixed.py
# Description  : Describe and control the general quantum state - mixed state.
#                Include two parts: density of states and super operator.
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>

from typing import List, Optional, Tuple, Union

from qusim.theory.hilbert import Hibert
from qusim.theory.operator import MechanicalOperator
from qusim.theory.qobj import Qobj
from qusim.theory.state import Bra, Ket, State
from qusim.theory.type import MatData


class DensityMatrix(Qobj):
    """
    Density matrix of pure or mixed states.

    Parameters
    ----------
    states : State of tuple
        State vectors.
        If the parameter is tuple, 
        the second number in the tuple represents the probability.

    """
    __type__ = "Density Matrix"

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(self, *states: Union[State, Tuple[State, float]]) -> None:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def inner(self, other: "DensityMatrix") -> float:
        """
        Calculate the overlap degree of two density matrices,
        defined by `trace(psi*phi)`.

        Parameters
        ----------
        other : DensityMatrix
            Another density matrix.

        Returns
        -------
        result : float
            Overlap degree.

        """
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Matrix Transformation

    @property
    def ket(self) -> Ket:
        """
        Get the state vector as a ket.
        
        Notes
        -----
        For mixed states, only one formal ket can be obtained, 
        and cannot be normalized.

        """
        raise NotImplementedError

    @property
    def bra(self) -> Bra:
        """
        Get the state vector as a bra.
        
        Notes
        -----
        For mixed states, only one formal bra can be obtained, 
        and cannot be normalized.

        """
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

    @property
    def purity(self) -> float:
        """Purity of mixed state, defined by trace(rho^2)."""
        raise NotImplementedError

    @property
    def entropy(self) -> float:
        """Etropy of mixed state, defined by -trace(rho*log(rho))."""
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Measure

    def measure(self, basis: Optional[MatData] = None) -> "DensityMatrix":
        """
        The density matrix is measured 
        and randomly collapses to an eigenstate 
        according to a certain probability.

        Parameters
        ----------
        basis : MatData, optional
            A matrix composed of all eigenvalues, 
            default by the basis of Hibert space.

        Returns
        -------
        result : DensityMatrix
            The density matrix obtained from collapse.

        """
        raise NotImplementedError

    def expect(self, observation: MechanicalOperator) -> float:
        """
        Average value of mechanical quantity ensemble, 
        that is `trace(rho*A)`.

        Parameters
        ----------
        observation : MechanicalOperator
            Any mechanical operator of the system.

        Returns
        -------
        result : float
            Expected value of mechanical quantity.

        """
        raise NotImplementedError

    def reduce(self,
               base_space: Hibert,
               *,
               keep: Optional[List[int]] = None,
               remove: Optional[List[int]] = None) -> "DensityMatrix":
        """
        Return reduced density matrix.

        If you are only interested in a part of a large system (its subsystem),
        you can trace the rest to obtain the 
        reduced density matrix of the subsystem.

        Notes
        -----
        Parameters `keep` and `remove` have and can only have one.

        Parameters
        ----------
        base_space : Hibert
            The basic Hilbert space 
            that constitutes a large direct product space.
        keep : list of int, optional
            The indexes of subsystems that want to keep.
        remove : list of int, optional
            The indexes of subsystems that want to remove.

        Returns
        -------
        result : DensityMatrix
            Reduced density matrix.

        """
        raise NotImplementedError


class SuperOperator(Qobj):
    """
    A super operator consisting of a series of Kraus operators.

    Notes
    -----
    Most of the operations supported by the matrix 
    are not supported by super operator.

    Parameters
    ----------
    Kraus : MatData of tuple
        Datas of Kraus operators. 
        If the parameter is tuple, 
        the second number in the tuple represents the probability.
    dim : int, optional
        Dimension of Hilbert space.

    Examples
    --------
    Quantum transition:

    >>> p = 0.05
    >>> K0 = [[1, 0], [0, sqrt(1 - p)]]
    >>> K1 = [[0, sqrt(p)], [0, 0]]
    >>> K = SuperOperator(K0, K1)
    >>> K
    SuperOperator
    -------------
    K0: [[1, 0],
         [0, 0.97]]
    K1: [[0, 0.22],
         [0, 0]]

    Depolarization:

    >>> p = 0.05
    >>> K = SuperOperator((Sigma(0), 1 - p), 
    ...                   (Sigma(1), p / 3), 
    ...                   (Sigma(2), p / 3), 
    ...                   (Sigma(3), p / 3))
    >>> K
    SuperOperator
    -------------
    K0: [[0, 0.13],
         [0.13, 0]]
    K1: [[0, 0.22],
         [0, 0]]
    K2: [[0, -0.13j],
         [0.13j, 0]]
    K3: [[0.13, 0],
         [0, -0.13]]

    -------------

    """

    __type__ = "Super operator"

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(self, *Kraus: Union[MatData, Tuple[MatData, float]],
                 dim: Optional[int]) -> None:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Comparison

    def __eq__(self, other: "SuperOperator") -> bool:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Basic Operations In Linear Space

    def __add__(self, other: "SuperOperator") -> "SuperOperator":
        raise NotImplementedError

    def __sub__(self, other: "SuperOperator") -> "SuperOperator":
        raise NotImplementedError

    def __mul__(self, other: float) -> "SuperOperator":
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def evolve(self,
               density_matrix: DensityMatrix,
               time: Optional[float] = None) -> DensityMatrix:
        """
        The density matrix of the system 
        follows the master equation and evolves with time.

        Parameters
        ----------
        density_matrix : DensityMatrix
            Density matrix of the system.
        time : float, optional
            Evolution time. If time is not `None`, 
            a time evolution operator will be generated 
            according to the Hamiltonian and evolution time, 
            otherwise it means that it itself is a time evolution operator.

        Returns
        -------
        result : DensityMatrix
            New density matrix.

        """
        raise NotImplementedError
