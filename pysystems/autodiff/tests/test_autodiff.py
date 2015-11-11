# -*- coding: utf-8 -*-
#HACK
import sys
sys.path.append('C:\\Work\\pySystems\\pysystems')

import autodiff as d
#import autodiff.diffnumpy as dnp
from autodiff import diffnumpy as dnp
import numpy as np
import math

if __name__ == '__main__':
    a = d.DiffFloat(2., name='a')
    b = d.DiffFloat(3., name='b')
    
    c = a + b

    v = np.array([a,b,c])
    v2 = np.vdot(v, v)    
    
    
    @d.DiffFunction
    def foo(x1, x2):
        return x1+x2, x1*x2, math.sin(x1)
        
    @foo.derivative(0)
    def foo(x1, x2):
        return 1., x2, math.cos(x1)
        
    @foo.derivative(1)
    def foo(x1, x2):
        return 1., x1, 0.
        
    b, n, m = foo(a,b)