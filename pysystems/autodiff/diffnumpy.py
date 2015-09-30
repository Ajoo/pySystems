# -*- coding: utf-8 -*-
import numpy as np
from collections import defaultdict
import functools

from autodiff import *

class DiffUFunc(DiffFunction):
    def __init__(self, fun, *dfun):#, **kdfun):
        self.fun = fun
        #functools.update_wrapper(self, fun)
        
        self.set_derivative(*dfun)
        
    def __call__(self, *args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        #kwargvalues = {(kw, arg.value if arg isinstance(DiffObject) else arg for kw, arg in kwargs.items)}
        f = self.fun(*argvalues, **kwargs)#, **kwargvalues)
        d = {}        
        
        for i, arg in enumerate(args):
            if isinstance(arg, DiffObject):
                #TODO: should I check derivative is provided?
                #TODO: provide option for numerically computed derivative if not
                #   provided?
                darg = arg.chain(self.dfun[i](*argvalues, **kwargs))
                for k, v in darg.viewitems():
                    d[k] = v + d[k] if k in d else v

        #chain if DiffObjArg is not None
        #if DiffObjArg is None append derivative
        if d:
            return DiffObject(f, d)
        else:
            return f

add = DiffUFunc(np.add, \
    lambda x1, x2: 1., \
    lambda x1, x2: 1.)
  
def _index_derivative(index, ndim):
    if not isinstance(index, tuple):
        index = (index,)
    if Ellipsis in index:
        index += (slice(None),)*ndim
    return index

#TODO: Make ndarray a factory class to support numpy nd arrays of objects that 
#are not floats
#TODO: Implement parametrization 'diag', 'sym', 'asym', 'triu', 'trid'
_value_prop_names = ('flags', 'strides', 'data', 'size', 'itemsize', 'nbytes')
@passthrough_properties('value', _value_prop_names)
class DiffNDArray(DiffObject):
    def __init__(self, value, d=None, parametrization='full', name=None, base=None):
#        if value.dtype is np.float64:
#            raise ValueError("Type {0} not implemented as a differentiable ndarray object"\
#                .format(value.dtype))
        self.value = np.asarray(value)
        self.name = name
        if d is not None:        
            self.d = d
            #self.track()
        else:
            self.d = {}
            self.track()
            
        self.base = base
    
    def __repr__(self):
        if self.name:
            return self.name + '(' + str(self.value) +')'
        else:
            return 'd' + repr(self.value)
    
    def __str__(self):
        return str(self.value)
        
    def __unicode__(self):
        return unicode(self.value)
        
    def __nonzero__(self):
        return bool(self.value)
        
    def __len__(self):
        return len(self.value)
    
    def __getitem__(self, index):
        value = self.value[index]   
        d = {k: di[_index_derivative(index, k.ndim)] for k, di in self.d.viewitems()}

        return DiffObject(value, d, base=self)
    
    def __setitem__(self, index, value):
        if value is DiffObject:
            self.value[index] = value.value
            
            #derivatives both in self and value
            for k in self.d.viewkeys() & value.d.viewkeys():
                self.d[k][_index_derivative(index, k.ndim)] = value.d[k]
            #derivatives only in self
            for k in self.d.viewkeys() - value.d.viewkeys():
                self.d[k][_index_derivative(index, k.ndim)] = 0.
            #derivatives only in value
            for k in value.d.viewkeys() - self.d.viewkeys():
                self.d[k] = np.zeros(self.shape + k.shape)
                self.d[k][_index_derivative(index, k.ndim)] = value.d[k]
        else:
            self.value[index] = value
            
            for k in self.d.viewkeys():
                self.d[k][_index_derivative(index, k.ndim)] = 0.
    
    def __delitem__(self, index):
        del self.value[index]
        
        for k in self.d.viewkeys():
            del self.d[k][_index_derivative(index, k.ndim)]
    
#    def __getslice__(self, i, j): 
#        return self.__getitem__((slice(i, j),))
#        
#    def __setslice__(self, i, j, value):
#        self.__setitem__((slice(i, j),), value)
#        
#    def __delslice__(self, i, j):
#        self.__delitem__((slice(i, j),))
    def __add__(self, other):
        return add(self, other) #returns wrapped add ufunc

    def __sub__(self, other):
        return add(self, -other)
    
    def __mul__(self, other):
        return multiply(self, other) #returns wrapped add ufunc
        
    def __floordiv__(self, other):
        return NotImplemented
        
    def __mod__(self, other):
        return NotImplemented
        
    def __divmod__(self, other, modulo=None):
        return NotImplemented
        
    def __pow__(self, other): #modulo?
        return power(self, other)
        
    def __lshift__(self, other):
        return NotImplemented
    
    def __rshift__(self, other):
        return NotImplemented
        
    def __and__(self, other):
        return NotImplemented
        
    def __xor__(self, other):
        return NotImplemented
        
    def __or__(self, other):
        return NotImplemented
    
    def __div__(self, other):
        return divide(self, other)
        
    def __truediv__(self, other): #? Difference?
        return divide(self, other)
    
    def __iter__(self):
        raise NotImplementedError("__iter__ not yet implemented")
        
    @property
    def shape(self):
        return self.value.shape
    
    @property
    def ndim(self):
        return self.value.ndim
    
    def track(self, d_self=None):
        '''
        Toggles on tracking for this object
        (Off by default when derivative information is provided)
        '''
        if d_self is None:
            d_self = np.reshape(np.identity(self.value.size), self.value.shape*2)
        self.d[self] = d_self
            
    def derivative(self, wrt):
        if isinstance(wrt, DiffNDArray):
            dshape = self.shape + wrt.shape
        else:
            dshape = self.shape
        return self.d.get(wrt, np.zeros(dshape))
    
    def chain(self, df):
        ax = (range(-self.ndim,0),range(self.ndim)) #last ndims df * first ndims d
        d = {k: np.tensordot(df,dk,axes=ax) for k, dk in self.d.viewitems()}
        return d
        
DiffObject._types[np.ndarray] = DiffNDArray
#for now register numpy scalar float types
#TODO: implement specialized scalar types
DiffObject._types[np.float64] = DiffNDArray
DiffObject._types[np.float32] = DiffNDArray
DiffObject._types[np.float_] = DiffNDArray

#alias
ndarray = DiffNDArray    
    
class DiffScalar(DiffObject):
    pass



# TEST
if __name__ == '__main__':
        
    vdot = DiffFunction(np.vdot, lambda x, y: y, lambda x, y: x)
    dot = DiffFunction(np.dot, lambda x, y: np.outer(np.identity(3),y).reshape((3,3,3)), lambda x, y: x)
    
    x = DiffObject(3*np.identity(3),name='x')
    y = DiffObject(np.array([1.,2.,3.]),name='y')
    
    z = dot(x ,y)
    
    def foo(x,M=np.identity(3)):
        return vdot(dot(M,x),x)
