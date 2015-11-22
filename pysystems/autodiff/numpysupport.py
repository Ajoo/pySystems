import sys
import numpy as np
from . import autodiff
try:
    from itertools import imap as map
except ImportError:
    pass

def dfloat(value):
    try:
        dvalue = map(autodiff.DiffFloat, value)
        if isinstance(value, np.ndarray):
            return np.array(dvalue)
        else:
            return type(value)(dvalue)
    except:
        return autodiff.DiffFloat(value)
        
autodiff.dfloat = dfloat
sys.modules['autodiff'].dfloat = dfloat
del dfloat