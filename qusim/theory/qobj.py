# Author       : czy
# Date         : 2022-03-17 20:36:11
# LastEditTime : 2022-03-17 20:36:12
# FilePath     : /qusim/core/base/qobj.py
# Description  : Base class of objects in Hilbert space,
#                declared some public interfaces.
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>

from functools import cached_property
from typing import Optional

from qusim.theory.hilbert import Hilbert
from qusim.theory.matrix import Matrix
from qusim.theory.type import MatData, Shape


class Qobj(Matrix):
    """
    Objects in Hilbert space, including state vectors and operators.

    The further encapsulation of matrix objects 
    combines the characteristics of Hilbert space.

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

    """

    __type__ = "Qobj"

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(self,
                 data: MatData,
                 shape: Optional[Shape] = None,
                 H: Optional[Hilbert] = None) -> None:
        super().__init__(data, shape)

        self.space = H or Hilbert(self.dim)

    # ----------------------------------------------------------------------
    # Basic properties

    @property
    def space(self) -> Hilbert:
        """Get the Hilbert space of the object."""
        return self.__space

    @space.setter
    def space(self, H: Hilbert) -> None:
        """Set the Hilbert space of the object."""
        if H.dim != self.dim:
            raise ValueError("dimension mismatch")
        self.__space = H

        if self.shape[0] == self.dim:
            self.array = H.directions.array.T @ self.array
        else:
            self.array = self.array @ H.directions.array.conj()

    @cached_property
    def dim(self) -> int:
        """Dimension of Hilbert space."""
        return max(self.shape)

    @cached_property
    def type(self) -> str:
        """Type of object, such as `State` or `Operator`."""
        return self.__type__
