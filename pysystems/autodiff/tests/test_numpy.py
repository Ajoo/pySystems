# -*- coding: utf-8 -*-
import sys
sys.path.append('C:\\Work\\pySystems\\pysystems')

import numpy as np

import autodiff as d
import autodiff.numpy as dnp

# TEST
if __name__ == '__main__':
    a = dnp.darray(3*np.identity(3), name='a')
    b = dnp.darray(np.array([1.,2.,3.]), name='b')

    
    c = a + b

    
    
    vdot = d.DiffFunction(np.vdot, lambda x, y: y, lambda x, y: x)
    dot = d.DiffFunction(np.dot, lambda x, y: np.outer(np.identity(3),y).reshape((3,3,3)), lambda x, y: x)
    z = dot(a ,b)