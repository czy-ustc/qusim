# Author       : czy
# Date         : 2022-03-17 17:01:09
# LastEditTime : 2022-03-17 17:01:09
# FilePath     : /qusim/core/base/matrix.py
# Description  : Provide the basic matrix operation
#                required by quantum mechanics (matrix mechanics).
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>
# from typing import Dict, Iterator, List, Optional, Sequence, Tuple, Union
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
        raise NotImplementedError

    def __init__(self, data: MatData, shape: Optional[Shape] = None) -> None:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:
        raise NotImplementedError

    def __repr__(self) -> str:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Comparison

    def __eq__(self, other: "Matrix") -> bool:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Basic Operations In Linear Space

    def __add__(self, other: "Matrix") -> "Matrix":
        """Addition: only matrices with the same shape can be added."""
        raise NotImplementedError

    def __sub__(self, other: "Matrix") -> "Matrix":
        """Subtraction: only matrices with the same shape can be subtracted."""
        raise NotImplementedError

    def __mul__(self, other: complex) -> "Matrix":
        """Multiplication: Matrix number multiplication."""
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """Matrix multiplication."""
        raise NotImplementedError

    def kron(self, other: "Matrix") -> "Matrix":
        """Direct product."""
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Matrix Transformation

    @property
    def H(self) -> "Matrix":
        """Conjugate transpose."""
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Other Methods

    def nonzero(self) -> Iterator[Element]:
        """
        Returns the nonzero element of the matrix (as iterator).

        Yields
        ----------
        element : Element
            The position and value of nonzero elements of the matrix.
        """
        raise NotImplementedError
