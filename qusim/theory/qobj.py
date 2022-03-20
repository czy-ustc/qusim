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

from qusim.theory.hilbert import Hibert
from qusim.theory.matrix import Matrix
from qusim.theory.type import MatData


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
        `data` represents the nonzero element of the state. 
        In this case, the `dim` parameter must be provided.
        If `data` is Matrix, the data of state is `data.data`.
    dim : int, optional
        Dimension of Hilbert space.

    """

    # ----------------------------------------------------------------------
    # Constructors

    def __init__(self, data: MatData, dim: Optional[int] = None) -> None:
        raise NotImplementedError

    # ----------------------------------------------------------------------
    # Basic properties

    @property
    def space(self) -> Hibert:
        """Get the Hilbert space of the object."""
        raise NotImplementedError

    @space.setter
    def space(self, H: Hibert) -> None:
        """Set the Hilbert space of the object."""
        raise NotImplementedError

    @cached_property
    def type(self) -> str:
        """Type of object, such as `State` or `Operator`."""
        raise NotImplementedError
