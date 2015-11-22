import numpy as np

from ..autodiff import DiffObject
#from .diffnumpy import *

def afunction(fun):
    '''
    Decorator for piecewise constant functions. The wrapped function will 
    work on DiffObjects arguments by replacing these with their values.
    The output is the regular function output, not a DiffObject
    ''' 
    def wrapped(*args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        
        fun(*argvalues, **kwargs)
        
        keys = set().union(*[arg.d for arg in args if isinstance(arg, DiffObject)])
        try:        
            for k in keys:
                argds = [arg.d[k] if isinstance(arg, DiffObject) else arg for arg in args]
                fun(*argds, **kwargs)
        except KeyError:
            raise AssertionError
    
    return wrapped
    
assert_functions = [name for name in dir(np.testing) if name.startswith('assert_')]
for fname in assert_functions:
    globals()[fname] = afunction(getattr(np.testing, fname))
#[
#'almost_equal',
#'approx_equal',
#'array_almost_equal',
#'allclose',
#'array_almost_equal_nulp',
#'array_max_ulp',
#'array_equal
#]