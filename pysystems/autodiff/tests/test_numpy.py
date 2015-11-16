# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\\Work\\pySystems\\pysystems')

import numpy as np

import autodiff as d
import autodiff.numpy as dnp

norm1 = lambda x: np.sum(np.abs(x))
def dequal(x, y, tol=100*d.DiffFloat.eps):
    return norm1(x.value-y.value) < tol*x.size and \
        all([norm1(x.d[k]-y.d[k]) < tol*x.d[k].size for k in x.d.keys()])  

# TEST
if __name__ == '__main__':
    a = dnp.darray(3*np.identity(3), name='a')
    b = dnp.darray(np.array([1.,2.,3.]), name='b')

    
    c = a + b

    
    
    vdot = d.DiffFunction(np.vdot, lambda x, y: y, lambda x, y: x)
    dot = d.DiffFunction(np.dot, lambda x, y: np.outer(np.identity(3),y).reshape((3,3,3)), lambda x, y: x)
    z = dot(a ,b)