# Author       : czy
# Date         : 2022-03-17 23:21:15
# LastEditTime : 2022-03-17 23:21:15
# FilePath     : /qusim/core/base/state.py
# Description  : State vector, including ket and Bra.
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>

from functools import reduce
import random
from itertools import zip_longest
from typing import List, Optional, Union

import numpy as np
from qusim.theory.hilbert import Hilbert
from qusim.theory.qobj import Qobj
from qusim.theory.type import MatData, Shape, __eps__


class State(Qobj):
    """
    Vector in Hilbert space describing the state of quantum mechanical system.

    Generalized state vector, including ket and bra.

    Notes
    -----
    Except linear operations such as addition and number multiplication, 
    the state vector will be normalized automatically before any operation.

    Parameters
    ----------
    data : MatData
        Data of state (type must be complex).
        If `data` is dict or sequence of tuple, 
        `data` represents the nonzero element of the matrix. 
        In this case, the `shape` parameter must be provided 
        a sparse matrix whose shape is specified by `shape`.
        If `data` is Matrix, the data of matrix is `data.data`.
    shape : Shape, optional
        If provided, the shape of the matrix will be specified by `shape`.
        The `shape` must be like (1, n), (n, 1) or (n, n).
    H : Hilbert, optional
        Hilbert space.
    unit : bool, default True
        Whether normalization processing is performed.

    """

    __type__ = "State"

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(self,
                 data: MatData,
                 shape: Optional[Shape] = None,
                 H: Optional[Hilbert] = None,
                 unit: bool = True) -> None:
        if shape is not None and not (shape[0] == 1 or shape[1] == 1):
            raise ValueError("'shape' must be like (1, n) or (n, 1)")
        super().__init__(data, shape, H)

        if unit:
            self.unit()

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:

        def format_complex(num: complex) -> str:
            if num.imag:
                if num.real:
                    return f"+({num.real:.2g}{num.imag:+.2g}i)"
                if abs(num.imag) == 1:
                    return ("+" if num.imag > 0 else "-") + "i"
                return f"{num.imag:+.2g}i"
            if abs(num.real) == 1:
                return "+" if num.real > 0 else "-"
            return f"{num.real:+.2g}"

        self.unit()
        states = self.space.names
        # bra
        if (self.shape[0] == 1):
            amplitudes = (self.array @ self.space.directions.array).flatten()
            string = "".join([
                f"{format_complex(amplitudes[i])}<{states[i]}|"
                for i in range(self.dim) if abs(amplitudes[i]) > __eps__
            ])
        # ket
        else:
            amplitudes = (self.space.directions.array @ self.array).flatten()
            string = "".join([
                f"{format_complex(amplitudes[i])}|{states[i]}>"
                for i in range(self.dim) if abs(amplitudes[i]) > __eps__
            ])

        return string[1:] if string.startswith("+") else string

    def __repr__(self) -> str:
        self.unit()
        value = self.__str__()
        return f"{self.type}[{value}]"

    # ----------------------------------------------------------------------
    # Comparison

    def __eq__(self, other: "State") -> bool:
        """
        Compare two state vectors.

        According to the assumption of quantum mechanics, 
        for any nonzero complex `c`, `|n> ~ c|n>`.
        That is, only the direction of the state vector, 
        not its size, makes sense.
        
        Returns
        -------
        result : bool
            Return True if two state vectors is equal, 
            otherwise return False.

        """
        if self.dim != other.dim:
            return False

        res = []
        for src1, src2 in zip_longest(self.nonzero(),
                                      other.nonzero(),
                                      fillvalue=((-1, -1), 0)):
            if (src1[0] != src2[0]):
                return False
            res.append(src1[1] / src2[1])

        for i in range(len(res) - 1):
            if abs(res[i] - res[-1]) > __eps__:
                return False

        return True

    def __ne__(self, other: "State") -> bool:
        return not self.__eq__(other)

    # ----------------------------------------------------------------------
    # Basic Operations In Linear Space

    def __add__(self, other: "State") -> "State":
        """Addition: only matrices with the same shape can be added."""
        if (self.shape == other.shape):
            return self.__class__(self.array.__add__(other.array),
                                  H=self.space,
                                  unit=False)
        else:
            raise ValueError(
                "add: only matrices with the same shape can be added")

    def __sub__(self, other: "State") -> "State":
        """Subtraction: only matrices with the same shape can be subtracted."""
        if (self.shape == other.shape):
            return self.__class__(self.array.__sub__(other.array),
                                  H=self.space,
                                  unit=False)
        else:
            raise ValueError(
                "sub: only matrices with the same shape can be subtracted")

    def __mul__(self, other: complex) -> "State":
        """Multiplication: Matrix number multiplication."""
        return self.__class__(self.array.__mul__(other),
                              H=self.space,
                              unit=False)

    def __rmul__(self, other: complex) -> "State":
        """Multiplication: Matrix number multiplication."""
        return self.__class__(self.array.__mul__(other),
                              H=self.space,
                              unit=False)

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def __matmul__(self, other: Qobj) -> Union[Qobj, "State", complex]:
        """
        Matrix multiplication.

        Notes
        -----
        Normalize first.

        """
        if isinstance(other, State):
            if self.shape[1] != other.shape[0]:
                raise ValueError(
                    "input operand 1 has a mismatch in its core dimension 0")

            if self.shape[0] == 1:
                return complex(self.unit().array.__matmul__(
                    other.unit().array))
            else:
                return Qobj(self.unit().array.__matmul__(other.unit().array))
        else:
            return type(self)(self.unit().array.__matmul__(other.array))

    def __rmatmul__(self, other: Qobj) -> Union[Qobj, "State", complex]:
        """
        Matrix multiplication.

        Notes
        -----
        Normalize first.

        """
        if isinstance(other, State):
            if self.shape[0] != other.shape[1]:
                raise ValueError(
                    "input operand 1 has a mismatch in its core dimension 0")

            if self.shape[1] == 1:
                return complex(self.unit().array.__rmatmul__(
                    other.unit().array))
            else:
                return Qobj(self.unit().array.__rmatmul__(other.unit().array))
        else:
            return type(self)(self.unit(False).array.__rmatmul__(other.array))

    def inner(self, other: "State") -> float:
        """
        Calculate the overlap degree of two state vectors,
        defined by `|<psi|phi>|^2`.

        Parameters
        ----------
        other : State
            Another state vector.

        Returns
        -------
        result : float
            Overlap degree.

        """
        return pow(abs(self.bra @ other.ket), 2)

    @classmethod
    def kron(cls, states: List["State"]) -> "State":
        """Direct product."""
        row_count = len(np.unique([s.shape[0] for s in states]))
        col_count = len(np.unique([s.shape[1] for s in states]))
        if not (row_count == 1 or col_count == 1):
            raise ValueError("dimension mismatch")
        H = Hilbert.kron([s.space for s in states])
        data = reduce(lambda x, y: np.kron(x, y), [s.array for s in states])
        return State(data, H=H)

    # ----------------------------------------------------------------------
    # Matrix Transformation

    @property
    def ket(self) -> "State":
        """Get the state vector as a ket."""
        self.unit()
        # bra
        if (self.shape[0] == 1):
            return self.H
        else:
            return self

    @property
    def bra(self) -> "State":
        """Get the state vector as a bra."""
        self.unit()
        # ket
        if (self.shape[1] == 1):
            return self.H
        else:
            return self

    def unit(self, inplace: bool = True) -> "State":
        """
        Return normalized (unit) vector.

        Make the sum of the modulo squares of the coefficients 1.

        Parameters
        ----------
        inplace : bool, default True
            If False, return a copy. 
            Otherwise, do operation inplace and return None.

        Returns
        -------
        result : State
            Normalized State.

        """
        if hasattr(self, "__hasunit") and getattr(self, "__hasunit"):
            return self
        setattr(self, "__hasunit", True)

        length = self.norm(2)
        if abs(length - 1) < __eps__:
            return self
        if inplace:
            self.array.__imul__(1 / length)
            return self
        else:
            return 1 / length * self

    # ----------------------------------------------------------------------
    # Measure

    def measure(self, basis: Optional[Hilbert] = None) -> "State":
        """
        The state vector is measured 
        and randomly collapses to an eigenstate 
        according to a certain probability.

        Parameters
        ----------
        basis : Hilbert, optional
            Hilbert space, default by `self.space`.

        Returns
        -------
        result : State
            The state vector obtained from collapse.

        """
        space = basis or self.space
        basis = space.directions
        # bra
        if (self.shape[0] == 1):
            amplitudes = (self.array @ basis.array).flatten()
        # ket
        else:
            amplitudes = (basis.array @ self.array).flatten()

        index = 0
        rand = random.uniform(0, 1)
        for p in map(lambda x: pow(abs(x), 2), amplitudes):
            if rand < p:
                break
            rand -= p
            index += 1

        if (self.shape[0] == 1):
            return self.__class__(basis[index], (1, self.dim), H=space)
        else:
            return self.__class__(basis[index], (self.dim, 1), H=space)

    # ----------------------------------------------------------------------
    # Metric

    def norm(self, ord: Union[int, str] = 2) -> float:
        """
        Return norm for state.

        Parameters
        ----------
        ord : int or str, default 2
            Order of the norm.

        Returns
        -------
        result : float
            Norm of the state vector.

        """
        if isinstance(ord, int):
            return np.linalg.norm(self.array.flatten(), ord=ord)
        elif ord == "inf":
            return np.linalg.norm(self.array.flatten(), ord=np.inf)
        else:
            return np.linalg.norm(self.array.flatten(), ord=int(ord))


class Ket(State):
    """
    State vector of Hilbert space.

    Its data is saved in the form of column matrix.

    Notes
    -----
    The initialization data can be row matrix or column matrix. 
    Finally, the data will be saved in the form of column matrix.

    """

    __type__ = "Ket"

    def __init__(self,
                 data: MatData,
                 dim: Optional[int] = None,
                 H: Optional[Hilbert] = None,
                 unit: bool = True) -> None:
        if dim is not None:
            super().__init__(data, shape=(dim, 1), H=H, unit=unit)
        else:
            super().__init__(data, H=H, unit=unit)
            self.resize((self.dim, 1))

    @property
    def H(self) -> "Bra":
        return Bra(self.array.T.conj(), H=self.space)

    @property
    def ket(self) -> "Ket":
        """Get the state vector as a ket."""
        self.unit()
        return self

    @property
    def bra(self) -> "Bra":
        """Get the state vector as a bra."""
        return Bra(self.array.T.conj(), H=self.space)

    def __matmul__(self, other: Qobj) -> Union[Qobj, State, complex]:
        res = super().__matmul__(other)
        if (isinstance(res, State)):
            if res.shape[0] == 1:
                return Bra(res, H=self.space)
            else:
                return Ket(res, H=self.space)
        return res

    def __rmatmul__(self, other: Qobj) -> Union[Qobj, State, complex]:
        res = super().__rmatmul__(other)
        if (isinstance(res, State)):
            if res.shape[0] == 1:
                return Bra(res, H=self.space)
            else:
                return Ket(res, H=self.space)
        return res

    @classmethod
    def kron(cls, states: List["Ket"]) -> "Ket":
        """Direct product."""
        res = super().kron(states)
        return Ket(res.array, H=res.space)


class Bra(State):
    """
    State vector of dual Hilbert space.

    Its data is saved in the form of row matrix.

    Notes
    -----
    The initialization data can be row matrix or column matrix. 
    Finally, the data will be saved in the form of row matrix.

    """

    __type__ = "Bra"

    def __init__(self,
                 data: MatData,
                 dim: Optional[int] = None,
                 H: Optional[Hilbert] = None,
                 unit: bool = True) -> None:
        if dim is not None:
            super().__init__(data, shape=(1, dim), H=H, unit=unit)
        else:
            super().__init__(data, H=H, unit=unit)
            self.resize((1, self.dim))

    @property
    def H(self) -> "Ket":
        return Ket(self.array.T.conj(), H=self.space)

    @property
    def ket(self) -> "Ket":
        """Get the state vector as a ket."""
        return Ket(self.array.T.conj(), H=self.space)

    @property
    def bra(self) -> "Bra":
        """Get the state vector as a bra."""
        return self

    def __matmul__(self, other: Qobj) -> Union[Qobj, State, complex]:
        res = super().__matmul__(other)
        if (isinstance(res, State)):
            if res.shape[0] == 1:
                return Bra(res, H=self.space)
            else:
                return Ket(res, H=self.space)
        return res

    def __rmatmul__(self, other: Qobj) -> Union[Qobj, State, complex]:
        res = super().__rmatmul__(other)
        if (isinstance(res, State)):
            if res.shape[0] == 1:
                return Bra(res, H=self.space)
            else:
                return Ket(res, H=self.space)
        return res

    @classmethod
    def kron(cls, states: List["Bra"]) -> "Bra":
        """Direct product."""
        res = super().kron(states)
        return Bra(res.array, H=res.space)
