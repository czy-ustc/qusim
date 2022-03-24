# Author       : czy
# Date         : 2022-03-16 20:25:44
# LastEditTime : 2022-03-16 20:25:44
# FilePath     : /qusim/core/base/hilbert.py
# Description  : Constructing and manipulating Hilbert space.
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>

from functools import cached_property, reduce
from itertools import product
from typing import Dict, List, Optional, Tuple, Union

from qusim.theory.matrix import Matrix
from qusim.theory.type import MatData


class Hilbert(object):
    """
    Hilbert space opened by a set of named basis vectors.

    The basis vectors are composed of numbers by default, 
    such as |0>, ..., |n>, and their direct product state are like |00>, |01>.

    Notes
    -----
    The directions of the basis vectors are subject to ket.

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
    subspace : list, optional
        Subspaces that make up Hilbert direct product spaces.

    Examples
    --------
    Basic usage: 

    >>> H = Hilbert(["+", "-"])
    >>> H
    Hilbert[|+>: [1.+0.j 0.+0.j],
            |->: [0.+0.j 1.+0.j]]
    >>> H = Hilbert(2)
    >>> H
    Hilbert[|0>: [1.+0.j 0.+0.j],
            |1>: [0.+0.j 1.+0.j]]

    Direct product space:

    >>> H = Hilbert(["+", "-"], N=2)
    >>> H
    Hilbert[|++>: [1.+0.j 0.+0.j 0.+0.j 0.+0.j],
            |+->: [0.+0.j 1.+0.j 0.+0.j 0.+0.j],
            |-+>: [0.+0.j 0.+0.j 1.+0.j 0.+0.j],
            |-->: [0.+0.j 0.+0.j 0.+0.j 1.+0.j]]

    Other orthogonal basis vector systems:

    >>> val = 1 / math.sqrt(2)
    >>> H = Hilbert(["alpha", "beta"], [[val, val], [val, -val]])
    >>> H
    Hilbert[|alpha>: [0.70710678+0.j 0.70710678+0.j],
            |beta>: [ 0.70710678+0.j -0.70710678+0.j]]
    >>> H = Hilbert(basis={"alpha": [val, val], "beta": [val, -val]})
    >>> H
    Hilbert[|alpha>: [0.70710678+0.j 0.70710678+0.j],
            |beta>: [ 0.70710678+0.j -0.70710678+0.j]]

    """

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(
            self,
            names: Optional[Union[List[str], int]] = None,
            directions: Optional[MatData] = None,
            basis: Optional[Dict[str, List[complex]]] = None,
            N: int = 1,
            subspace: Optional[List[Tuple[List[str], Matrix]]] = None) -> None:
        if names is not None:

            if isinstance(names, int):
                names = [str(i) for i in range(names)]
            if directions is None:
                directions = Matrix.eye(len(names))
            self.__names = names
            self.__directions = Matrix(directions)

        elif basis is not None:
            self.__names = list(basis.keys())
            self.__directions = Matrix(list(basis.values()))

        else:
            raise ValueError(
                "need to pass in arguments `names` or argument `basis`")

        if len(self.__names) != self.__directions.shape[0]:
            raise ValueError("dimension mismatch")

        if (self.__directions.H @ self.__directions) != Matrix.eye(self.dim):
            raise ValueError(
                "the basis vectors have no orthogonally normalized")

        if N > 1:
            self.__names = [
                "".join(name) for name in product(self.__names, repeat=N)
            ]
            self.__directions = reduce(lambda x, y: x.kron(y),
                                       [self.__directions] * N)

        if subspace is None:
            self.__subspace = [(self.names, self.directions)]
        else:
            self.__subspace = subspace

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:
        basis = f",\n ".join(
            [f"|{k}>: {v}" for k, v in zip(self.names, self.directions)])
        return f"[{basis}]"

    def __repr__(self) -> str:
        name = self.__class__.__name__
        basis = f",\n{' ' * (len(name) + 1)}".join(
            [f"|{k}>: {v}" for k, v in zip(self.names, self.directions)])
        return f"{name}[{basis}]"

    # ----------------------------------------------------------------------
    # Comparison

    def __eq__(self, other: "Hilbert") -> bool:
        """
        Notes
        -----
        Compare only `directions`, not `names`.

        """
        return self.directions == other.directions

    def __ne__(self, other: "Hilbert") -> bool:
        return not self.__eq__(other)

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def __mul__(self, other: Union["Hilbert", int]) -> "Hilbert":
        """
        Direct product of Hilbert space.

        Parameters
        ----------
        other : Hilbert or int
            Direct product with `other` if `other` is a Hilbert space, 
            else generate `dim` oplus `other` direct product space.

        Returns
        -------
        space : Hilbert
            A new Hilbert space.

        """
        if isinstance(other, int):
            return self.kron([self] * other)
        else:
            return self.kron([self, other])

    @classmethod
    def kron(cls, spaces: List["Hilbert"]) -> "Hilbert":
        """Direct product of Hilbert space."""
        names = reduce(lambda x, y: ["".join(name) for name in product(x, y)],
                       [H.names for H in spaces])
        directions = reduce(lambda x, y: x.kron(y),
                            [H.directions for H in spaces])
        subspace = reduce(lambda x, y: x + y, [H.subspace for H in spaces])
        return Hilbert(names=names, directions=directions, subspace=subspace)

    # ----------------------------------------------------------------------
    # Basic Properties

    @cached_property
    def dim(self) -> int:
        """Dimension of Hilbert space."""
        return len(self.__names)

    @property
    def names(self) -> List[str]:
        """Get the name of basis vector."""
        return self.__names

    @names.setter
    def names(self, basis_vector: Union[List[str], int]) -> None:
        """Set the name of basis vector."""
        if isinstance(basis_vector, int):
            basis_vector = [str(i) for i in range(basis_vector)]

        if len(basis_vector) != self.dim:
            raise ValueError("dimension mismatch")

        self.__names = basis_vector

    @property
    def directions(self) -> Matrix:
        """Get the direction of basis vector."""
        return self.__directions

    @directions.setter
    def directions(self, basis_vector: Matrix) -> None:
        """Set the direction of basis vector."""
        if self.__directions.shape != basis_vector.shape:
            raise ValueError("dimension mismatch")
        if (basis_vector.H @ basis_vector) != Matrix.eye(self.dim):
            raise ValueError(
                "the basis vectors have no orthogonally normalized")

        self.__directions = basis_vector

    @property
    def basis(self) -> Dict[str, List[complex]]:
        """Get the basis vector."""
        return {k: Matrix(v) for k, v in zip(self.names, self.directions)}

    @basis.setter
    def basis(self, basis_vector: Dict[str, List[complex]]) -> None:
        """Set the basis vector."""
        names = list(basis_vector.keys())
        directions = Matrix(list(basis_vector.values()))

        if (directions.H @ directions) != Matrix.eye(self.dim):
            raise ValueError(
                "the basis vectors have no orthogonally normalized")

        self.__names = names
        self.__directions = directions

    @property
    def subspace(self) -> List[Tuple[List[str], Matrix]]:
        return self.__subspace
