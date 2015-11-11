# -*- coding: utf-8 -*-
import numpy as np
import itertools as it
import functools as ft
from collections import defaultdict


from ..autodiff import *

__all__ = [
    'DiffUFunc',
    
    'DiffNDArray',
    'darray'
    ]
def passthrough_properties(field, prop_names):
    def decorate(cls):
        for prop_name in prop_names:
            #fix prop_name
            def getter(self, prop_name=prop_name):
                return getattr(getattr(self, field), prop_name)
            
            def setter(self, value, prop_name=prop_name):
                return setattr(getattr(self, field), prop_name, value)
                
            def deleter(self, prop_name=prop_name):
                return delattr(getattr(self, field), prop_name)
            
            #? maybe partial prop_name out?
            setattr(cls, prop_name, property(getter, setter, deleter))
        
        return cls
    return decorate

def _broadcast_derivative(index, dfun):
    def new_dfun(*args, **kwargs):
        result = dfun(*args, **kwargs)
        
        rshape = result.shape
        dshape = arg[index].shape
        
        rndim = len(rshape)
        dndim = len(dshape)
        
        no_broadcast = [dim > 1 for dim in dshape]
        idx1 = np.array([False]*(rndim-dndim) + no_broadcast + [False]*dndim)
        idx2 = np.array([False]*rndim + no_broadcast)
        
        a = np.arange(rndim+dndim)
        a[idx1], a[idx2] = a[idx2], a[idx1]
        
        result = result[[Ellipsis]+[np.newaxis]*dndim].transpose(a)
        return np.broadcast_to(result, rshape+dshape)
    return new_dfun

_fun_prop_names = ('nin', 'nout', 'nargs', 'ntypes', 'types', 'identity')
@passthrough_properties('fun', _fun_prop_names)
class DiffUFunc(DiffFunction):
    def set_derivatives(self, *dfun):
        self.dfun = list(it.starmap(_broadcast_derivative, enumerate(dfun)))
    
    def set_derivative(self, index, dfun):
        dfun = _broadcast_derivative(index, dfun)
        if index < len(self.dfun):
            self.dfun[index] = dfun

        self.dfun += [NoneFunction]*(index-len(self.dfun)) + [dfun]


    
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
_value_prop_names = ('shape', 'ndim', 'flags', 'strides', 'data', 'size', \
    'itemsize', 'nbytes')
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
DiffObject._types[np.float16] = DiffNDArray  
    
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

