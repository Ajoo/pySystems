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

import types
import functools
import numbers
import math
#import operator


__all__ = [
    'DiffObject',
    
    'DiffFunction',
    'dfunction',
    
    'ConstFunction',
    'cfunction',
    
    'DiffFloat',
    'dfloat'
    ]
    
#? Maybe keep track of created objects?
#? maybe replace with factory FUNCTION?
class DiffObject(object):
    '''
    Class that acts as a factory for the different Differentible Object types.
    It uses the type of the value passed to figure out which descendent to 
    instantiate based on the types registered in the static dict _types
    '''
    _types = {}
    #TODO: DiffList and DiffTuple and DiffDict
    def __new__(cls, value, *args, **kwargs):
        if cls is DiffObject:
            try:
                dsc = DiffObject._types[type(value)]
            except:
                raise TypeError("Type {0} not implemented as a differentiable object"\
                    .format(type(value)))
            return super(DiffObject, cls).__new__(dsc)
        
        return super(DiffObject, cls).__new__(cls)
            
        
    def __init__(self, value, d=None, d_self=1.0):
        self.value = value
        if d is not None:        
            self.d = d
        else:
            self.d = {self:d_self}
            
    #? change d to be dictionary specialization that implements chain method?
    def chain(self, df):
        raise NotImplementedError("Chain method not implemented for DiffObject")
    
    def derivative(self, wrt):
        return NotImplemented
        
    def D(self, wrt):
        return self.derivative(wrt)
    
    def __hash__(self):
        return id(self)

#TODO
class DiffDict(DiffObject):
    pass

     
#TODO: work out mechanism to write derivatives wrt to an object that contains
#   DiffObject attributes
#TODO: define differentiable methods that provide derivatives wrt self (a DiffClass)
#   This should have a convinient way to define method and dmethods
class DiffClass(DiffObject):
    pass


def _not_implemented_func(*args, **kwargs):
    return NotImplemented
NoneFunction = _not_implemented_func

def sum_dicts(*dlist):
    d = {}
    for di in dlist:
        for k, v in di.viewitems():
            d[k] = v + d.get(k, 0.) #v + d[k] if k in d else v
    return d

def is_dobject(x):
    return isinstance(x,DiffObject)


class DiffFunction(object):
    def __init__(self, fun, *dfun):#, **kdfun):
        self.fun = fun
        #functools.update_wrapper(self, fun)
        #self.__name__ = fun.__name__
        #self.__doc__ = fun.__doc__
        
        self.set_derivatives(*dfun)
        #self.kdfun = kdfun
    
    def set_derivatives(self, *dfun):
        self.dfun = list(dfun)
    
    def set_derivative(self, index, dfun):
        if index < len(self.dfun):
            self.dfun[index] = dfun

        self.dfun += [NoneFunction]*(index-len(self.dfun)) + [dfun]
           
    def derivative(self, index=0):
        '''
        Provides a convenient way to define derivatives by decorating functions
        as @function.derivative(index)
        '''
        def derivative_decorator(dfun):
            self.set_derivative(index, dfun)
            return self
        return derivative_decorator
        
    def __get__(self, instance, cls):
        #return functools.partial(DiffFunction.__call__, self, instance) 
        return types.MethodType(self, instance)
        
    def finite_differences(self, *args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs #TODO: for now can not diff wrt kwargs       
        
        f = self.fun(*argvalues, **kwargvalues)
        if not any([isinstance(arg, DiffObject) for arg in args]):
            return f
        
        df = [self.finite_difference(i, arg, f, *argvalues, **kwargvalues) \
            for i, arg in enumerate(args) if isinstance(arg, DiffObject)]
        if type(f) in DiffObject._types:        
            d = sum_dicts(*df)
            return DiffObject(f, d)
        else:
            d = [sum_dicts(*d) for d in zip(*df)]
            return type(f)(map(DiffObject, f, d))

    
    def finite_difference(self, index, darg, f, *args, **kwargs):
        farg = lambda arg: self.fun(*(args[0:index]+(arg,)+args[index+1:]), **kwargs)
        d = map(farg, darg.delta())
        if type(f) in DiffObject._types:
            return darg.chain_from_delta(f, d)
        elif isinstance(f, Iterable):
            return [darg.chain_from_delta(fi, di) for fi, di in zip(f,zip(*d))]
            
        raise TypeError('DiffFunction output not implemented as a DiffObject')
    
    fd = finite_differences #alias for convenience
    
    #TODO: use itertools for lazy evaluation and memory efficiency
    def __call__(self, *args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs #TODO: for now can not diff wrt kwargs       
        
        #? should I check is all derivatives are provided?
        #? provide option for numerically computed derivative if not defined?
        f = self.fun(*argvalues, **kwargvalues)
        
        if not any([isinstance(arg, DiffObject) for arg in args]):
            return f
        if self.dfun:
            #compute df_args
            df = [self.dfun[i](*argvalues, **kwargvalues) \
                 if isinstance(arg, DiffObject) else None \
                 for i, arg in enumerate(args)]
        else:
            #if self.dfun is empty assume fun returns a tuple of nominal 
            #value and derivative list
            f, df = f
            
        #try to make DiffObject
        if type(f) in DiffObject._types:
            dlist = [arg.chain(dfi) for arg, dfi in zip(args, df) if isinstance(arg, DiffObject)]
            d = sum_dicts(*dlist)
            return DiffObject(f, d)
        elif isinstance(f, Iterable):
            dlist = [[arg.chain(dfij) for dfij in dfi] for arg, dfi in zip(args, df) if isinstance(arg, DiffObject)]
            d = [sum_dicts(*d) for d in zip(*dlist)]
            return type(f)(map(DiffObject, f, d))
            
        raise TypeError('DiffFunction output not implemented as a DiffObject')

#class DiffMethod(DiffFunction):    
#    #Makes DiffMethod a non-data descriptor that binds its __call__ method to
#    #the particular instance that calls it
#    def __get__(self, instance, cls):
#        return functools.partial(DiffFunction.__call__, self, instance) 
#        #return types.MethodType(self, instance)

dfunction = DiffFunction #alias
##HACK
#class DictWrapper(object):
#    def __init__(self,d):
#        self.__dict__ = d
##? Alternative implementation?
#def dfunction(fun, *dfun):
#    self = DictWrapper(locals())
#    
#    def wrapped(*args, **kwargs):
#        return DiffFunction.__call__.im_func(self, *args, **kwargs)
#    try:
#        return functools.wraps(fun)(wrapped)
#    except:
#        return wrapped

class ConstFunction(object):
    def __init__(self, fun, *dfun):#, **kdfun):
        self.fun = fun
    
    def __call__(self, *args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs #TODO: for now can not diff wrt kwargs
        
        return self.fun(*argvalues, **kwargvalues)
        
    def __get__(self, instance, cls):
        #return functools.partial(DiffFunction.__call__, self, instance) 
        return types.MethodType(self, instance)
        
#function wrapper for piecewise constant functions
#the wrapped function will return its normal output (not a DiffObject) with
#all arguments replaced by their values in case these
def cfunction(fun):
    '''
    Decorator for piecewise constant functions. The wrapped function will 
    work on DiffObjects arguments by replacing these with their values.
    The output is the regular function output, not a DiffObject
    ''' 
    def wrapped(*args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs #TODO: for now can not diff wrt kwargs
        
        return fun(*argvalues, **kwargvalues)
    return wrapped
#------------------------------------------------------------------------------
#   DiffFloat - DiffObject specialization for representing floats with
#               derivative information
#? Generalize to Complex? DiffNumber or DiffScalar?
class DiffFloat(DiffObject):
    eps = 1e-8
    def __init__(self, value, d=None, name=None):
        if not isinstance(value, numbers.Number):
            raise ValueError("DiffFloat does not support {0} values".format(type(value)))
        self.value = value
        self.name = name
        if d is not None:        
            self.d = d
            #self.track()
        else:
            self.d = {self: 1.}
    
    def track(self, d_self=None):
        '''
        Toggles on tracking for this object
        (Off by default when derivative information is provided)
        '''
        if d_self is None:
            d_self = 1.
        self.d[self] = d_self
            
    def derivative(self, wrt):
        return self.d.get(wrt, 0.)
    
    def chain(self, df):
        d = {k: df*dk for k, dk in self.d.viewitems()}
        return d
  
    def delta(self, eps=None):
        if not eps:
            eps = self.eps
        return [self.value + eps]
    
    def chain_from_delta(self, f, delta, eps=None):
        if not eps:
            eps = self.eps
        df = (delta[0]-f)/eps
        return self.chain(df)
    
    def __repr__(self):
        if self.name:
            return self.name + '(' + repr(self.value) + ')'
        else:
            return 'dfloat(' + repr(self.value) + ')'
    
    def __str__(self):
        return str(self.value)
        
    def __unicode__(self):
        return unicode(self.value)
        
    def __bool__(self):
        return bool(self.value)
        
    def conjugate(self):
        d = {k: v.conjugate() for k, v in self.d.viewitems()}
        return DiffFloat(self.value.conjugate(), d)
    
    @property
    def imag(self):
        d = {k: v.imag for k, v in self.d.viewitems()}
        return DiffFloat(self.value.imag, d)
    
    @property
    def real(self):
        d = {k: v.real for k, v in self.d.viewitems()}
        return DiffFloat(self.value.real, d)
    
    #python 2
    __nonzero__ = __bool__
        
DiffObject._types[float] = DiffFloat

#Factory function for DiffFloat container objects
def dfloat(value):
    try:
        return type(value)(map(DiffFloat, value))
    except:
        return DiffFloat(value)

#float operations not implemented (yet)
ops_not_implemented = ('mod', 'divmod', 'floordiv', 'trunc')
    
#for opname in ops_not_implemented:
#    setattr(DiffFloat, '__{0}__'.format(opname), _not_implemented_func)

ops_act_on_value = ('lt', 'le', 'eq', 'ne', 'ge', 'gt', 'int', 'float')

for op_sname in ops_act_on_value + ops_not_implemented:
    op_lname = '__{0}__'.format(op_sname)
    setattr(DiffFloat, op_lname, cfunction(getattr(float, op_lname)))

#unary operations
def sign(x):
    if x > 0:
        return 1.
    elif x < 0:
        return -1.
    elif x == 0:
        return 0.

dirac_1arg = lambda x: 0. if x != 0. else float('inf')
zero_1arg = lambda x: 0.

uops_derivatives = {
    #dabs returns 0 as a subgradient at 0
    'abs': (sign,),
    'neg': (lambda x: -1.,),
    'pos': (lambda x: 1.,),
}

#binary operations   
bops_derivatives = {
    'add': (lambda x, y: 1.,)*2,
    'sub': (lambda x, y: 1., lambda x, y: -1.),
    'mul': (lambda x, y: y, lambda x, y: x),
    'div': (lambda x, y: 1./y, lambda x, y: -x/y**2),
    'truediv': (lambda x, y: 1./y, lambda x, y: -x/y**2),
    'pow': (lambda x, y: y*x**(y-1.), lambda x, y: math.log(x)*x**y)
}
#TODO mod, floordiv

#add reflected binary ops
def reflected(fun):
    return (lambda x, y: fun(y, x))
    
bops_derivatives.update({'r' + opname: map(reflected, dop) \
    for opname, dop in bops_derivatives.viewitems()})

#add unary and binary ops to DiffFloat dict
for op_sname, dops in uops_derivatives.items() + bops_derivatives.items():
    op_lname = '__{0}__'.format(op_sname)
    setattr(DiffFloat, op_lname, DiffFunction(getattr(float, op_lname), *dops))
    #could also use DiffMethod
    
if __name__ == '__main__':
    import numpy as np    
    
    a = DiffFloat(2., name='a')
    b = DiffFloat(3., name='b')
    
    c = a + b

    v = np.array([a,b,c])
    v2 = np.vdot(v, v)    
    
    
    @DiffFunction
    def foo(x1, x2):
        return x1+x2, x1*x2, math.sin(x1)
        
    @foo.derivative(0)
    def foo(x1, x2):
        return 1., x2, math.cos(x1)
        
    @foo.derivative(1)
    def foo(x1, x2):
        return 1., x1, 0.
        
    d, n, m = foo(a,b)
    
    print('Ops not in float: ', set(dir(a))-set(dir(1.)))
    print()
    print('Ops not in dfloat: ', set(dir(1.))-set(dir(a)))