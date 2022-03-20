# Author       : czy
# Date         : 2022-03-18 10:18:43
# LastEditTime : 2022-03-18 10:18:43
# FilePath     : /qusim/core/base/type.py
# Description  : Alias of base type.
#
# Copyright 2022 Zhiyuan Chen <chenzhiyuan@mail.ustc.edu.cn>

from typing import Dict, List, NewType, Sequence, Tuple, Union

import numpy as np

# One or two dimensional complex matrix.
ListData = Union[List[complex], List[List[complex]]]

# A dict of nonzero elements of matrix.
DictData = Dict[Tuple[int, int], complex]

# A array-like object of nonzero elements of matrix.
SeqData = Sequence[Tuple[Tuple[int, int]], complex]

# Basic matrix type.
Matrix = NewType("Matrix", np.ndarray)

# Type of parameters that make up the matrix.
MatData = Union[ListData, DictData, SeqData, Matrix]

# Shape of matrix.
Shape = Tuple[int, int]

# Matrix element position.
Index = Tuple[int, int]

# Matrix element.
Element = Tuple[Index, complex]
