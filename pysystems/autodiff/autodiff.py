# -*- coding: utf-8 -*-
from collections import defaultdict
import functools
import numbers

#Maybe keep track of created objects?
#TODO: add factory FUNCTION for convenience
class DiffObject(object):
    '''
    Class that acts as a factory for the different Differentible Object types.
    It uses the type of the value passed to figure out which descendent to 
    instantiate based on the types registered in the static dict _types
    '''
    _types = {}
    #TODO: DiffList and DiffTuple
    def __new__(cls, value, *args, **kwargs):
        print str(cls)
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
            
    def chain(self, df):
        raise ValueError("Chain method not implemented for DiffObject")
    
    def __hash__(self):
        return id(self)
        

#TODO: if only fun is passed, assume it returns fun, dfun, kdfun
class DiffFunction(object):
    
    def __init__(self, fun, *dfun):#, **kdfun):
        self.fun = fun
        functools.update_wrapper(self, fun)
        
        self.set_derivative(*dfun)
        #self.kdfun = kdfun
    
    def set_derivative(self, *dfun):
        self.dfun = dfun
    
    def __call__(self, *args, **kwargs):
        argvalues = [arg.value if isinstance(arg, DiffObject) else arg for arg in args]
        #kwargvalues = {(kw, arg.value if arg isinstance(DiffObject) else arg for kw, arg in kwargs.items)}
        f = self.fun(*argvalues, **kwargs)#, **kwargvalues)
        d = {}        
        
        for i, arg in enumerate(args):
            if isinstance(arg, DiffObject):
                darg = arg.chain(self.dfun[i](*argvalues, **kwargs))
                for k, v in darg.viewitems():
                    d[k] = v + d[k] if k in d else v

        #chain if DiffObjArg is not None
        #if DiffObjArg is None append derivative
        if d:
            return DiffObject(f, d)
        else:
            return f
        
