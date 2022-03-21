# Author       : czy
# Date         : 2022-03-17 17:01:09
# LastEditTime : 2022-03-17 17:01:09
# FilePath     : /qusim/core/base/matrix.py
# Description  : Provide the basic matrix operation
#                required by quantum mechanics (matrix mechanics).
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>
# from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
from collections.abc import Iterable
from itertools import product
from typing import Iterator, Optional, Union

import numpy as np
from qusim.theory.type import Element, MatData, Shape


class Matrix(np.ndarray):
    """
    Matrix object based on the principle of matrix mechanics.

    The behavior of object `np.ndarray` is modified 
    to conform to the operation rules of matrix mechanics.

    Parameters
    ----------
    data : MatData
        Data of matrix (type must be complex).
        If `data` is dict or sequence of tuple, 
        `data` represents the nonzero element of the matrix. 
        In this case, the `shape` parameter must be provided 
        a sparse matrix whose shape is specified by `shape`.
        If `data` is Matrix, the data of matrix is `data.data`.
    shape : Shape, optional
        If provided, the shape of the matrix will be specified by `shape`.
        The `shape` must be like (1, n), (n, 1) or (n, n).

    Examples
    --------
    Generate matrix from list:

    >>> mat = Matrix([1, 0])
    >>> mat
    Matrix([[1, 2]])
    >>> mat = Matrix([1, 0], (2, 1))
    >>> mat
    array([[1],
           [2]])

    Generating matrices from nonzero elements:

    >>> mat = Matrix({(0, 0): 1, (1, 1): 1}, (2, 2))
    >>> mat
    Matrix([[1, 0],
            [0, 1]])
    >>> mat = Matrix([((0, 0), 1)], (1, 3))
    >>> mat
    Matrix([[1, 0, 0]])

    """

    __eps__ = 1e-12

    # ----------------------------------------------------------------------
    # Constructors

    def __new__(cls,
                data: MatData,
                shape: Optional[Shape] = None) -> Union["Matrix", complex]:
        """
        Notes
        -----
        If the matrix has only one element,
        it will be converted to `complex`.

        """

        if shape is not None and not (shape[0] == shape[1] or shape[0] == 1
                                      or shape[1] == 1):
            raise ValueError("'shape' must be like (1, n), (n, 1) or (n, n)")

        number = (int, float, complex)

        # MatData
        if isinstance(data, (Matrix, np.ndarray)):
            shape = data.shape if len(data.shape) == 2 else (1, data.shape[0])
            instance = super(Matrix, cls).__new__(cls,
                                                  shape=shape,
                                                  buffer=data,
                                                  dtype=complex)

        # ListData
        elif isinstance(data, list) and isinstance(data[0], (list, *number)):

            if isinstance(data[0], number):
                data = [data]
            instance = super(Matrix,
                             cls).__new__(cls,
                                          shape=(len(data), len(data[0])),
                                          buffer=np.array(data, dtype=complex),
                                          dtype=complex)
            if shape is not None:
                instance = instance.reshape(shape)

        # DictData or SeqData
        elif isinstance(data, (dict, Iterable)):

            instance = super(Matrix, cls).__new__(cls,
                                                  shape=shape,
                                                  dtype=complex)
            instance.fill(0)
            iter = data.items() if isinstance(data, dict) else data
            for k, v in iter:
                instance[k] = v

        if instance.shape == (1, 1):
            return instance[(0, 0)]

        return instance

    @classmethod
    def eye(cls, N: int) -> "Matrix":
        """Create an identity matrix of `N * N`."""
        return Matrix(np.eye(N, dtype=complex))

    @classmethod
    def ones(cls, N: int) -> "Matrix":
        """Create an all one matrix of `N * N`."""
        return Matrix(np.ones((N, N), dtype=complex))

    @classmethod
    def zeros(cls, N: int) -> "Matrix":
        """Create an all zero matrix of `N * N`."""
        return Matrix(np.zeros((N, N), dtype=complex))

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:
        return super().__str__()

    def __repr__(self) -> str:
        return super().__repr__()

    # ----------------------------------------------------------------------
    # Comparison

    def __eq__(self, other: "Matrix") -> bool:
        if self.shape != other.shape:
            return False

        return all(
            abs(self[p] - other[p]) < self.__eps__
            for p in product(range(self.shape[0]), range(self.shape[1])))

    def __ne__(self, other: "Matrix") -> bool:
        return not self.__eq__(other)

    # ----------------------------------------------------------------------
    # Basic Operations In Linear Space

    def __add__(self, other: "Matrix") -> "Matrix":
        """Addition: only matrices with the same shape can be added."""
        if (self.shape == other.shape):
            return super().__add__(other)
        else:
            raise ValueError(
                "add: only matrices with the same shape can be added")

    def __sub__(self, other: "Matrix") -> "Matrix":
        """Subtraction: only matrices with the same shape can be subtracted."""
        if (self.shape == other.shape):
            return super().__sub__(other)
        else:
            raise ValueError(
                "sub: only matrices with the same shape can be subtracted")

    def __mul__(self, other: complex) -> "Matrix":
        """Multiplication: Matrix number multiplication."""
        return super().__mul__(other)

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """Matrix multiplication."""
        return Matrix(super().__matmul__(other))

    def kron(self, other: "Matrix") -> "Matrix":
        """Direct product."""
        return np.kron(self, other)

    # ----------------------------------------------------------------------
    # Matrix Transformation

    @property
    def H(self) -> "Matrix":
        """Conjugate transpose."""
        return self.T.conj()

    # ----------------------------------------------------------------------
    # Other Methods

    def nonzero(self) -> Iterator[Element]:
        """
        Returns the nonzero element of the matrix (as iterator).

        Notes
        -----
        The judgment standard of nonzero element is 
        whether the absolute value of matrix element is less than `__eps__`.

        Yields
        ----------
        element : Element
            The position and value of nonzero elements of the matrix.
        """

        return filter(
            lambda x: abs(x[1]) > self.__eps__,
            ((p, self[p])
             for p in product(range(self.shape[0]), range(self.shape[1]))))
