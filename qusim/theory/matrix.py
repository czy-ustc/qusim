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
from typing import Iterator, Optional, Tuple, Union

import numpy as np
from qusim.theory.type import Element, MatData, Shape, __eps__


class Matrix(object):
    """
    Matrix object based on the principle of matrix mechanics.

    The behavior of object `np.ndarray` is modified 
    to conform to the operation rules of matrix mechanics.

    Notes
    -----
    `Matrix` must be two-dimensional, that is, 
    its `shape` must be a tuple with a length of 2.

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
                shape: Optional[Shape] = None,
                *args,
                **kwargs):
        """
        Notes
        -----
        If the matrix has only one element,
        it will be converted to `complex`.

        """
        # MatData
        if isinstance(data, (Matrix, np.ndarray)):
            if data.size == 1:
                return data[(0, 0)]
        # ListData
        elif isinstance(data, list) and isinstance(
                data[0], (list, int, float, complex)):
            if len(data) == 1:
                if not isinstance(data[0], list):
                    return data[0]
                elif len(data[0]) == 1:
                    return data[0][0]
        # DictData
        elif isinstance(data, dict):
            if shape[0] * shape[1] == 1:
                return complex(data[(0, 0)])
        # SeqData
        elif isinstance(data, Iterable):
            if shape[0] * shape[1] == 1:
                return complex(data[0][1])

        return super().__new__(cls)

    def __init__(self, data: MatData, shape: Optional[Shape] = None) -> None:
        if shape is not None and not (shape[0] == shape[1] or shape[0] == 1
                                      or shape[1] == 1):
            raise ValueError("'shape' must be like (1, n), (n, 1) or (n, n)")

        number = (int, float, complex)

        # MatData
        if isinstance(data, (Matrix, np.ndarray)):
            shape = shape or (data.shape if len(data.shape) == 2 else
                              (1, data.shape[0]))
            self.__array = np.array(data.data, dtype=complex)
            self.__array.resize(shape)

        # ListData
        elif isinstance(data, list) and isinstance(data[0], (list, *number)):

            if isinstance(data[0], number):
                data = [data]
            self.__array = np.array(data, dtype=complex)
            if shape is not None:
                self.__array.resize(shape)

        # DictData or SeqData
        elif isinstance(data, (dict, Iterable)):

            self.__array = np.zeros(shape=shape, dtype=complex)
            iter = data.items() if isinstance(data, dict) else data
            for k, v in iter:
                self.__array[k] = v

    @classmethod
    def eye(cls, N: int) -> "Matrix":
        """Create an identity matrix of `N * N`."""
        return Matrix(np.eye(N, dtype=complex))

    @classmethod
    def ones(cls, shape: Union[int, Shape]) -> "Matrix":
        """
        Create an all one matrix.

        Parameters
        ----------
        shape : int or Shape
            Shape of matrix.
            If `shape` is int, 
            it represents the square matrix of `shape * shape`.

        """
        if isinstance(shape, int):
            return Matrix(np.ones((shape, shape), dtype=complex))
        else:
            if shape is not None and not (shape[0] == shape[1] or shape[0] == 1
                                          or shape[1] == 1):
                raise ValueError(
                    "'shape' must be like (1, n), (n, 1) or (n, n)")
            return Matrix(np.ones(shape, dtype=complex))

    @classmethod
    def zeros(cls, shape: Union[int, Shape]) -> "Matrix":
        """
        Create an all zero matrix.

        Parameters
        ----------
        shape : int or Shape
            Shape of matrix.
            If `shape` is int, 
            it represents the square matrix of `shape * shape`.

        """
        if isinstance(shape, int):
            return Matrix(np.zeros((shape, shape), dtype=complex))
        else:
            if shape is not None and not (shape[0] == shape[1] or shape[0] == 1
                                          or shape[1] == 1):
                raise ValueError(
                    "'shape' must be like (1, n), (n, 1) or (n, n)")
            return Matrix(np.zeros(shape, dtype=complex))

    # ----------------------------------------------------------------------
    # Formatting

    def __str__(self) -> str:
        return self.__array.__str__()

    def __repr__(self) -> str:
        raw_str = self.__array.__repr__()
        raw_name = "array"
        new_name = self.__class__.__name__
        new_str = raw_str.replace(raw_name,
                                  new_name).replace(" " * (len(raw_name) + 1),
                                                    " " * (len(new_name) + 1))
        return new_str

    # ----------------------------------------------------------------------
    # Comparison

    def __eq__(self, other: "Matrix") -> bool:
        if self.shape != other.shape:
            return False

        return all(
            abs(self[p] - other[p]) < __eps__
            for p in product(range(self.shape[0]), range(self.shape[1])))

    def __ne__(self, other: "Matrix") -> bool:
        return not self.__eq__(other)

    # ----------------------------------------------------------------------
    # Basic Operations In Linear Space

    def __add__(self, other: "Matrix") -> "Matrix":
        """Addition: only matrices with the same shape can be added."""
        if (self.shape == other.shape):
            return self.__class__(self.__array.__add__(other.__array))
        else:
            raise ValueError(
                "add: only matrices with the same shape can be added")

    def __sub__(self, other: "Matrix") -> "Matrix":
        """Subtraction: only matrices with the same shape can be subtracted."""
        if (self.shape == other.shape):
            return self.__class__(self.__array.__sub__(other.__array))
        else:
            raise ValueError(
                "sub: only matrices with the same shape can be subtracted")

    def __mul__(self, other: complex) -> "Matrix":
        """Multiplication: Matrix number multiplication."""
        return self.__class__(self.__array.__mul__(other))

    def __rmul__(self, other: complex) -> "Matrix":
        """Multiplication: Matrix number multiplication."""
        return self.__class__(self.__array.__mul__(other))

    # ----------------------------------------------------------------------
    # Multiplication Operations

    def __matmul__(self, other: "Matrix") -> "Matrix":
        """Matrix multiplication."""
        return self.__class__(self.__array.__matmul__(other.__array))

    def kron(self, other: "Matrix") -> "Matrix":
        """Direct product."""
        return self.__class__(np.kron(self.__array, other.__array))

    # ----------------------------------------------------------------------
    # Matrix Transformation

    @property
    def T(self) -> "Matrix":
        """Transpose matrix."""
        return self.__class__(self.__array.T)

    @property
    def conj(self) -> "Matrix":
        """Conjugate matrix."""
        return self.__class__(self.__array.conj())

    @property
    def H(self) -> "Matrix":
        """Conjugate transpose."""
        return self.__class__(self.__array.T.conj())

    def resize(self, shape: Shape) -> None:
        self.__array.resize(shape)

    def reshape(self, shape: Shape) -> "Matrix":
        return self.__class__(self.__array.reshape(shape))

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
            lambda x: abs(x[1]) > __eps__,
            ((p, self[p])
             for p in product(range(self.shape[0]), range(self.shape[1]))))

    # ----------------------------------------------------------------------
    # Basic Properties

    @property
    def shape(self) -> Shape:
        return self.__array.shape

    @property
    def size(self) -> int:
        return self.__array.size

    @property
    def data(self) -> memoryview:
        return self.__array.data

    @property
    def array(self) -> np.ndarray:
        return self.__array

    @array.setter
    def array(self, mat: Union["Matrix", np.ndarray]) -> None:
        if self.shape != mat.shape:
            raise ValueError("dimension mismatch")
        if isinstance(mat, Matrix):
            self.__array = mat.array
        else:
            self.__array = mat

    # ----------------------------------------------------------------------
    # Container Methods

    def __getitem__(
            self, key: Union[int, Tuple[int,
                                        int]]) -> Union[np.ndarray, complex]:
        return self.__array.__getitem__(key)

    def __setitem__(self, key: Tuple[int, int], value: complex) -> None:
        self.__array.__setitem__(key, value)

    # ----------------------------------------------------------------------
    # Conversion

    def toarray(self) -> np.ndarray:
        return self.__array

    def tolist(self) -> list:
        return self.__array.tolist()

    def tobytes(self) -> bytes:
        return self.__array.tobytes()
