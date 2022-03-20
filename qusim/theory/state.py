# Author       : czy
# Date         : 2022-03-17 23:21:15
# LastEditTime : 2022-03-17 23:21:15
# FilePath     : /qusim/core/base/state.py
# Description  : State vector, including ket and Bra.
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>

from ctypes import Union
from typing import Optional

from qusim.theory.qobj import Qobj
from qusim.theory.type import MatData


class State(Qobj):
    """
    Vector in Hilbert space describing the state of quantum mechanical system.

    Generalized state vector, including ket and bra.

    """

    __type__ = "State"

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
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def __matmul__(self, other: "State") -> "State":
        """
        Matrix multiplication.

        Notes
        -----
        Normalize first.

        """
        raise NotImplementedError

    def dot(self, other: "State") -> complex:
        """
        Dot multiplication.
        
        Notes
        -----
        The state vector is calculated as a vector.

        """
        raise NotImplementedError

    def cross(self, other: "State") -> "State":
        """
        Cross multiplication.

        Notes
        -----
        The state vector is calculated as a vector.

        """
        raise NotImplementedError

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
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Matrix Transformation

    @property
    def H(self) -> "State":
        """
        Return adjoint (dagger) of state.

        Change `|n>`(`<n|`) into `<n|`(`|n>`).

        Notes
        -----
        The type of state vector will change, 
        that is, ket becomes bra and bra becomes ket.

        Returns
        -------
        result : State
            Normalized State.
        """
        raise NotImplementedError

    @property
    def ket(self) -> "State":
        """Get the state vector as a ket."""
        raise NotImplementedError

    @property
    def bra(self) -> "State":
        """Get the state vector as a bra."""
        raise NotImplementedError

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
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Measure

    def measure(self, basis: Optional[MatData] = None) -> "State":
        """
        The state vector is measured 
        and randomly collapses to an eigenstate 
        according to a certain probability.

        Parameters
        ----------
        basis : MatData, optional
            A matrix composed of all eigenvalues, 
            default by the basis of Hibert space.

        Returns
        -------
        result : State
            The state vector obtained from collapse.

        """
        raise NotImplementedError

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
        raise NotImplementedError


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

    def __init__(self, data: MatData, dim: Optional[int] = None) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError


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

    def __init__(self, data: MatData, dim: Optional[int] = None) -> None:
        raise NotImplementedError

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError
