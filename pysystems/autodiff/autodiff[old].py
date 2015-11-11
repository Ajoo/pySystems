# -*- coding: utf-8 -*-
from __future__ import division

from collections import defaultdict, Iterable

import functools
import numbers
import math
#import operator
import numpy as np

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

#Maybe keep track of created objects?
#TODO: add factory FUNCTION for convenience
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
        else:
            return super(DiffObject, cls).__new__(cls)
            
        
    def __init__(self, value, d=None, d_self=1.0):
        self.value = value
        if d is not None:        
            self.d = d
        else:
            self.d = {self:d_self}
            
    #? change d to be dictionary specialization that implements chain method?
    def chain(self, df):
        raise ValueError("Chain method not implemented for DiffObject")
    
    def __hash__(self):
        return id(self)
        
#Factory function for DiffObject types
def dobject(value, d=None):
    try: #? if isinstance(value, Iterable):
        dvalue = map(DiffObject, value, d)
        print dvalue
        if isinstance(value, np.ndarray):
            return np.array(dvalue)
        else:
            return type(value)(dvalue)
    except:
        return DiffObject(value, d)

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



#TODO: if only fun is passed, assume it returns fun, dfun, kdfun
class DiffFunction(object):
    def __init__(self, fun, *dfun):#, **kdfun):
        self.fun = fun
        #functools.update_wrapper(self, fun)
        
        self.set_derivatives(*dfun)
        #self.kdfun = kdfun
    
    def set_derivatives(self, *dfun):
        self.dfun = list(dfun)
    
    def set_derivative(self, index, dfun):
        if index < len(self.dfun):
            self.dfun[index] = dfun
        else:
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
        
    def __call__(self, *args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs #TODO: for now can not diff wrt kwargs       
        
        if self.dfun:
            f = self.fun(*argvalues, **kwargvalues)#, **kwargvalues)
        else:
            #if self.dfun is empty assume fun returns a tuple of nominal 
            #value and derivative list
            f, df = self.fun(*argvalues, **kwargvalues)
        
        #TODO: list or tuple output checks
        d = {}
        for i, arg in enumerate(args):
            if isinstance(arg, DiffObject):
                #TODO: should I check derivative is provided?
                #TODO: provide option for numerically computed derivative if not
                #   provided?
                if self.dfun:
                    darg = arg.chain(self.dfun[i](*argvalues, **kwargvalues))
                else:
                    darg = arg.chain(df[i])
                    
                for k, v in darg.viewitems():
                    d[k] = v + d[k] if k in d else v

        #chain if DiffObjArg is not None
        #if DiffObjArg is None append derivative
        return dobject(f, d)

def dfunction(fun, *dfun):
    def wrapped(*args, **kwargs):
        print args
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        kwargvalues = kwargs #TODO: for now can not diff wrt kwargs

        f = fun(*argvalues, **kwargvalues)#, **kwargvalues)
        #TODO: support enumerable outputs
        d = {}        
        
        for i, arg in enumerate(args):
            if isinstance(arg, DiffObject):
                #TODO: should I check derivative is provided?
                #TODO: provide option for numerically computed derivative if not
                #   provided?
                darg = arg.chain(dfun[i](*argvalues, **kwargvalues))
                for k, v in darg.viewitems():
                    d[k] = v + d[k] if k in d else v

        #chain if DiffObjArg is not None
        #if DiffObjArg is None append derivative
        return dobject(f, d)
    
    return wrapped

#class DictWrapper(object):
#    def __init__(self,d):
#        self.__dict__ = d
##? Alternative implementation?
#def dfunction(fun, *dfun):
#    self = DictWrapper(locals())
#    def wrapped(*args, **kwargs):
#        print args
#        return DiffFunction.__call__.im_func(self, *args, **kwargs)
#    return wrapped
#    #return functools.partial(DiffFunction.__call__.im_func, DictWrapper(locals()))

#function wrapper for piecewise constant functions
#the wrapped function will return its normal output (not a DiffObject) with
#all arguments replaced by their values in case these
def pwcfunction(fun):
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
cfunction = pwcfunction        
        

#TODO: descriptor version of DiffFunction for use in wrapping class methods
class DiffMethod(DiffFunction):    
    #Make DiffMethod a non-data descriptor that binds its __call__ method to
    #the particular instance that calls it
    def __get__(self, instance, cls):
        return functools.partial(DiffFunction.__call__, self, instance) 
    


#------------------------------------------------------------------------------
#   DiffFloat - DiffObject specialization for representing floats with
#               derivative information
#? Generalize to Complex?
class DiffFloat(DiffObject):
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
  
    def __repr__(self):
        if self.name:
            return self.name + '(' + repr(self.value) + ')'
        else:
            return 'dfloat(' + repr(self.value) + ')'
    
    def __str__(self):
        return str(self.value)
        
    def __unicode__(self):
        return unicode(self.value)
        
    #for numpy:
    def sqrt(self):
        return self**.5
        
DiffObject._types[float] = DiffFloat

#Factory function for DiffFloat container objects
#? Should it work with numpy.ndarray? or leave that to diffnumpy?
def dfloat(value):
    try:
        dvalue = map(DiffFloat, value)
        print dvalue
        if isinstance(value, np.ndarray):
            return np.array(dvalue)
        else:
            return type(value)(dvalue)
    except:
        return DiffFloat(value)

#float operations not implemented (yet)
ops_not_implemented = ('mod', 'divmod', 'floordiv', 'trunc')
    
for opname in ops_not_implemented:
    setattr(DiffFloat, '__{0}__'.format(opname), _not_implemented_func)

ops_act_on_value = ('lt', 'le', 'eq', 'ne', 'ge', 'gt', 'nonzero',\
    'int', 'float')

for op_sname in ops_act_on_value:
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
    setattr(DiffFloat, op_lname, dfunction(getattr(float, op_lname), *dops))
    #could also use DiffMethod
    
if __name__ == '__main__':
    a = DiffFloat(2., name='a')
    b = DiffFloat(3., name='b')
    
    c = a + b

    v = np.array([a,b,c])
    v2 = np.vdot(v, v)    