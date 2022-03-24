# Author       : czy
# Date         : 2022-03-18 20:23:22
# LastEditTime : 2022-03-18 20:28:08
# FilePath     : /qusim/theory/mixed.py
# Description  : Describe and control the general quantum state - mixed state.
#                Include two parts: density of states and super operator.
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>

import math
from functools import cached_property, reduce
from itertools import product
from typing import Any, List, Optional, Tuple, Union

import numpy as np
from qusim.theory.hilbert import Hilbert
from qusim.theory.matrix import Matrix
from qusim.theory.operator import Operator
from qusim.theory.qobj import Qobj
from qusim.theory.state import Bra, Ket, State
from qusim.theory.type import MatData
from scipy.linalg import logm


class DensityMatrix(Qobj):
    """
    Density matrix of pure or mixed states.

    Parameters
    ----------
    states : State or tuple
        State vectors.
        If the parameter is tuple, 
        the second number in the tuple represents the probability.
    data : MatData, optional
        Data of density matrix.
    dim : int, optional
        Dimension of density matrix.
    H : Hilbert, optional
        Hilbert space.

    """
    __type__ = "Density Matrix"

    # ----------------------------------------------------------------------
    # Constructors

    def __new__(cls, *args, **kwargs) -> Any:
        if "data" in kwargs:
            if "dim" in kwargs:
                shape = (kwargs["dim"], kwargs["dim"])
            else:
                shape = None
            return super().__new__(cls,
                                   data=kwargs.pop("data"),
                                   shape=shape,
                                   *args,
                                   **kwargs)
        else:
            if isinstance(args[0], tuple):
                dim = args[0][0].dim
            else:
                dim = args[0].dim
            return super().__new__(cls, Matrix.eye(dim), *args, **kwargs)

    def __init__(self,
                 *states: Union[State, Tuple[State, float]],
                 data: Optional[MatData] = None,
                 dim: Optional[int] = None,
                 H: Optional[Hilbert] = None) -> None:
        if data is None:
            probability_sum = sum(state[1] if isinstance(state, tuple) else 1
                                  for state in states)
            array = [
                (s[0].ket.array @ s[0].bra.array) *
                s[1] if isinstance(s, tuple) else s.ket.array @ s.bra.array
                for s in states
            ]
            data = np.sum(array, axis=0) / probability_sum
            super().__init__(data, H=H)
        else:
            if dim is None:
                super().__init__(data, H=H)
            else:
                super().__init__(data, shape=(dim, dim), H=H)

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

    def __add__(self, other: Qobj) -> Qobj:
        return NotImplemented

    def __sub__(self, other: Qobj) -> Qobj:
        return NotImplemented

    def __mul__(self, other: complex) -> Qobj:
        return NotImplemented

    def __rmul__(self, other: complex) -> Qobj:
        return NotImplemented

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def overlap(self, other: "DensityMatrix") -> float:
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
        return (self.array @ other.array).trace()

    # ----------------------------------------------------------------------
    # Matrix Transformation

    @cached_property
    def ket(self) -> Ket:
        """
        Get the state vector as a ket.
        
        Notes
        -----
        For mixed states, only one formal ket can be obtained, 
        and cannot be normalized.

        """
        a0 = math.sqrt(self[(0, 0)].real)
        data = [complex(i / a0).conjugate() for i in self[0]]
        return Ket(data, H=self.space, unit=False)

    @cached_property
    def bra(self) -> Bra:
        """
        Get the state vector as a bra.
        
        Notes
        -----
        For mixed states, only one formal bra can be obtained, 
        and cannot be normalized.

        """
        a0 = math.sqrt(self[(0, 0)].real)
        data = [i / a0 for i in self[0]]
        return Bra(data, H=self.space, unit=False)

    # ----------------------------------------------------------------------
    # Metric

    @cached_property
    def trace(self) -> complex:
        """Trace of matrix."""
        return self.array.trace()

    @cached_property
    def det(self) -> float:
        """Determinant of matrix."""
        return np.linalg.det(self.array)

    @cached_property
    def purity(self) -> float:
        """Purity of mixed state, defined by trace(rho^2)."""
        return abs(np.trace(self.array @ self.array))

    # ----------------------------------------------------------------------
    # Kronecker Product And Partial Trace

    @classmethod
    def kron(cls, dms: List["DensityMatrix"]) -> "DensityMatrix":
        """Direct product."""
        H = Hilbert.kron([dm.space for dm in dms])
        data = reduce(lambda x, y: np.kron(x, y), [dm.array for dm in dms])
        return DensityMatrix(data=data, H=H)

    def ptrace(self, sel: Union[List[int], int]) -> "DensityMatrix":
        """
        Partial trace of the quantum object.

        If you are only interested in a part of a large system (its subsystem),
        you can trace the rest to obtain the 
        reduced density matrix of the subsystem.

        Parameters
        ----------
        sel : list of int or int
            An int or list of components to keep after partial trace. 
            The order is unimportant; 
            no transposition will be done 
            and the spaces will remain in the same order in the output.

        Returns
        -------
        result : DensityMatrix
            Reduced density matrix.

        """

        if isinstance(sel, int):
            sel = {sel}
        else:
            sel = set(sel)

        subspace = self.space.subspace
        count = len(subspace)
        removes = list(set(range(count)) - sel)
        removes.sort()

        dim = self.dim // reduce(lambda x, y: x * y,
                                 (subspace[i][1].shape[0] for i in removes))

        density = np.zeros((dim, dim), dtype=complex)
        for arrays in product(*[
                subspace[i][1] if i in
                removes else [np.eye(subspace[i][1].shape[0])]
                for i in range(count)
        ]):
            left = reduce(lambda x, y: np.kron(x, y), arrays)
            right = left.T.conj()
            density += left @ self.array @ right

        space = Hilbert.kron([Hilbert(*subspace[i]) for i in list(sel)])
        return DensityMatrix(data=density, H=space)


class SuperOperator(Qobj):
    """
    A super operator consisting of a series of Kraus operators.

    Notes
    -----
    Most of the operations supported by the matrix 
    are not supported by super operator.

    Parameters
    ----------
    Kraus : MatData or tuple
        Datas of Kraus operators. 
        If the parameter is tuple, 
        the second number in the tuple represents the probability.
    dim : int, optional
        Dimension of Hilbert space.
    H : Hilbert, optional
        Hilbert space.

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
    K0: [[1.  +0.j 0.  +0.j]
        [0.  +0.j 0.97+0.j]]
    K1: [[0.  +0.j 0.22+0.j]
        [0.  +0.j 0.  +0.j]]

    Depolarization:

    >>> p = 0.05
    >>> K = SuperOperator((Sigma(0), 1 - p), 
    ...                   (Sigma(1), p / 3), 
    ...                   (Sigma(2), p / 3), 
    ...                   (Sigma(3), p / 3))
    >>> K
    SuperOperator
    -------------
    K0: [[0.97+0.j 0.  +0.j]
        [0.  +0.j 0.97+0.j]]
    K1: [[0.  +0.j 0.13+0.j]
        [0.13+0.j 0.  +0.j]]
    K2: [[0.+0.j   0.-0.13j]
        [0.+0.13j 0.+0.j  ]]
    K3: [[ 0.13+0.j  0.  +0.j]
        [ 0.  +0.j -0.13+0.j]]

    -------------

    """

    __type__ = "Super operator"

    # ----------------------------------------------------------------------
    # Constructors

    def __new__(cls, *args, **kwargs) -> Any:
        if isinstance(args[0], tuple):
            dim = Operator(args[0][0], dim=kwargs.get("dim", None)).dim
        else:
            dim = Operator(args[0], dim=kwargs.get("dim", None)).dim
        return super().__new__(cls, Matrix.eye(dim), *args, **kwargs)

    def __init__(self,
                 *Kraus: Union[MatData, Tuple[MatData, float]],
                 dim: Optional[int] = None,
                 H: Optional[Hilbert] = None) -> None:

        if isinstance(Kraus[0], tuple):
            pro_sum = sum(k[1] for k in Kraus)
            self.__operators = [
                Operator(Matrix(k[0]).array * math.sqrt(k[1] / pro_sum),
                         dim=dim,
                         H=H) for k in Kraus
            ]
        else:
            self.__operators = [
                Operator(Matrix(k), dim=dim, H=H) for k in Kraus
            ]

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:
        name = self.__class__.__name__
        separator = "-" * len(name)
        count = len(self.operators)
        with np.printoptions(precision=2):
            return "\n".join([separator] + [
                f"K{i}: " +
                str(self.operators[i].array).replace("\n", "\n    ")
                for i in range(count)
            ])

    def __repr__(self) -> str:
        name = self.__class__.__name__
        return f"{name}\n{self.__str__()}"

    # ----------------------------------------------------------------------
    # Comparison

    def __eq__(self, other: "SuperOperator") -> bool:
        return self.operators == other.operators

    def __ne__(self, other: "SuperOperator") -> bool:
        return not self.__eq__(other)

    # ----------------------------------------------------------------------
    # Basic Operations In Linear Space

    def __add__(self, other: Qobj) -> Qobj:
        return NotImplemented

    def __sub__(self, other: Qobj) -> Qobj:
        return NotImplemented

    def __mul__(self, other: complex) -> Qobj:
        return NotImplemented

    def __rmul__(self, other: complex) -> Qobj:
        return NotImplemented

    # ----------------------------------------------------------------------
    # Basic Properties

    @property
    def operators(self) -> List[Operator]:
        return self.__operators
