# -*- coding: utf-8 -*-
# Scalar version of numpy support
# Doesn't define DiffNDArray objects but instead relies on standard
# numpy.ndarrays of DiffFloat Objects

import numpy as np
from .. import autodiff

##################### DiffFloat Mods ##########################################
#Overwrite dfloat to work with numpy arrays
def dfloat(value):
    try:
        dvalue = map(autodiff.DiffFloat, value)
        print(dvalue)
        if isinstance(value, np.ndarray):
            return np.array(dvalue)
        else:
            return type(value)(dvalue)
    except:
        return autodiff.DiffFloat(value)

autodiff.dfloat = dfloat
#Monkey patch DiffFloat with numpy specific attributes
autodiff.DiffFloat.sqrt = lambda self: self**.5
autodiff.DiffFloat.ndim = 0
autodiff.DiffFloat.shape = tuple()