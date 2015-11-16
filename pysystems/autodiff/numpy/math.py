# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from collections import defaultdict, Iterable
try:
    from itertools import izip as zip
    from itertools import imap as map
    from itertools import ifilter as filter
except ImportError:
    pass

import numpy as np
from ..autodiff import cfunction
from .diffnumpy import DiffUFunc, DiffNDArray

##############################  MATH  OPS  ####################################
cufuncs = ['floor_divide', 'remainder', 'mod', 'fmod', 'rint', 'sign', 'ones_like']
for ufunc_name in cufuncs:
    globals()[ufunc_name] = cfunction(getattr(np, ufunc_name))

ones2args = lambda x1, x2: np.ones(np.broadcast(x1, x2).shape)

ufunc_derivatives = {
    'negative': [lambda x: -np.ones_like(x)],
    'absolute': [np.sign],
    'exp': [np.exp],
    'exp2': [lambda x: np.log(2)*np.exp2(x)],
    'log': [np.reciprocal],
    'log2': [lambda x: np.reciprocal(x)/np.log(2)],
    'log10': [lambda x: np.reciprocal(x)/np.log(10)],
    'expm1': [np.exp],
    'log1p': [lambda x: np.reciprocal(1.+x)],
    'sqrt': [lambda x: 0.5*x**-.5],
    'square': [lambda x: 2*x],
    'reciprocal': [lambda x: -np.reciprocal(x**2)],
    'add': [ones2args]*2,
    'subtract': [lambda x, y: -ones2args]*2,
    'multiply': [lambda x, y: np.broadcast_arrays(x, y)[1],
                 lambda x, y: np.broadcast_arrays(x, y)[0]],
    'true_divide': [lambda x, y: np.reciprocal(y),
                    lambda x, y: -x/y**2],
    'power': [lambda x, y: y*x**(y-1.),
              lambda x, y: np.log(x)*x**y]
}
ufunc_derivatives['divide'] = ufunc_derivatives['true_divide']

for ufunc_name, dufunc in ufunc_derivatives.viewitems():
    globals()[ufunc_name] = DiffUFunc(getattr(np, ufunc_name), *dufunc)

#unary ops
uops = {
    'abs': absolute,
    'neg': negative
}
#binary ops
bops = {
    'add': add,
    'sub': subtract,
    'mul': multiply,
    'div': divide,
    'truediv': true_divide
}
#add reflected binary ops
def reflected(fun):
    return (lambda x, y: fun(y, x))
bops.update({'r' + opname: reflected(op) for opname, op in bops.viewitems()})

for op_sname, ufunc in dict(uops, **bops).viewitems():
    op_lname = '__{0}__'.format(op_sname)
    setattr(DiffNDArray, op_lname, ufunc)

__all__ = cufuncs + ufunc_derivatives.keys()
del ones2args, cufuncs, ufunc_derivatives
del uops, bops, reflected
##############################  TRIG FUNCS  ###################################