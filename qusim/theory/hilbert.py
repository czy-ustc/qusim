# Author       : czy
# Date         : 2022-03-16 20:25:44
# LastEditTime : 2022-03-16 20:25:44
# FilePath     : /qusim/core/base/hilbert.py
# Description  : Constructing and manipulating Hilbert space.
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>

from functools import cached_property
from typing import Dict, List, Optional, Union

from qusim.theory.matrix import Matrix
from qusim.theory.type import MatData


class Hibert(object):
    """
    Hilbert space opened by a set of named basis vectors.

    The basis vectors are composed of numbers by default, 
    such as |0>, ..., |n>, and their direct product state are like |00>, |01>.

    Parameters
    ----------
    names : list of str or int, optional
        Names of basis vectors. 
        If `names` is int, the names will be generated as follows: 
        [str(i) for i in range(int)].
    directions : MatData, optional
        Unitary matrix, default to Identity matrix (I). 
        Indicate the direction of the basis vector 
        (each row of the matrix corresponds to a basis vector).
    basis : dict, optional
        The keys of `basis` will be the names of basis vectors,
        and the values of `basis` will be the directions of basis vectors.
    N : int, default 1
        Generate `dim` oplus `N` direct product space 
        as the new Hilbert space if N > 1.

    Examples
    --------
    Basic usage: 

    >>> H = Hibert(["+", "-"])
    >>> H
    Hibert[|+>: [1, 0],
           |->: [0, 1]]
    >>> H = Hibert(2)
    >>> H
    Hibert[|0>: [1, 0],
           |1>: [0, 1]]

    Direct product space:

    >>> H = Hibert(["+", "-"], N=2)
    >>> H
    Hibert[|++>, |+->, |-+>, |-->]
    Hibert[|++>: [1, 0, 0, 0],
           |+->: [0, 1, 0, 0],
           |-+>: [0, 0, 1, 0],
           |-->: [0, 0, 0, 1]]

    Other orthogonal basis vector systems:

    >>> H = Hibert(["alpha", "beta"], [[0.707, 0.707], [0.707, -0.707]])
    >>> H
    Hibert[|alpha>: [0.707, 0.707],
           |beta>: [0.707, -0.707]]
    >>> H = Hibert(basis={"alpha": [0.707, 0.707], "beta": [0.707, -0.707]})
    >>> H
    Hibert[|alpha>: [0.707, 0.707],
           |beta>: [0.707, -0.707]]

    """

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
        self,
        names: Optional[Union[List[str], int]] = None,
        directions: Optional[MatData] = None,
        basis: Optional[Dict[str, List[complex]]] = None,
        N: int = 1,
    ) -> None:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def __mul__(self, other: Union["Hibert", int]) -> "Hibert":
        """
        Direct product of Hilbert space.

        Parameters
        ----------
        other : Hibert or int
            Direct product with `other` if `other` is a Hibert space, 
            else generate `dim` oplus `other` direct product space.

        Returns
        -------
        space : Hibert
            A new Hibert space.

        """
        raise NotImplementedError

    @classmethod
    def kron(cls, spaces: List["Hibert"]) -> "Hibert":
        """Direct product of Hilbert space."""
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Basic Properties

    @cached_property
    def dim(self) -> int:
        """Dimension of Hilbert space."""
        raise NotImplementedError

    @property
    def names(self) -> List[str]:
        """Get the name of basis vector."""
        raise NotImplementedError

    @names.setter
    def names(self, basis_vector: List[str]) -> None:
        """Set the name of basis vector."""
        raise NotImplementedError

    @property
    def directions(self) -> Matrix:
        """Get the direction of basis vector."""
        raise NotImplementedError

    @directions.setter
    def directions(self, basis_vector: Matrix) -> None:
        """Set the direction of basis vector."""
        raise NotImplementedError

    @property
    def basis(self) -> Dict[str, List[complex]]:
        """Get the basis vector."""
        raise NotImplementedError

    @basis.setter
    def basis(self, basis_vector: Dict[str, List[complex]]) -> None:
        """Set the basis vector."""
        raise NotImplementedError
