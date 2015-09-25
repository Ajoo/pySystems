from abc import ABCMeta, abstractmethod

import numpy as np
from scipy.signal import lfilter
import matplotlib.pyplot as plt
#import scipy.signal.ltisys, scipy.signal.dltisys

class DynamicalSystem(object):
    """
    The DynamicalSystem class serves as the base for all Dynamical System
    specializations and defines their basic interface
    """
    __metaclass__ = ABCMeta

    def __init__(self):
        pass
    
    @abstractmethod
    def sim(self, **kwargs):
        pass

class LTI(DynamicalSystem):
    
    __metaclass__ = ABCMeta



class DStateSpace(LTI):
    """
    Deterministic discrete-time State Space System
    """
    def __init__(self, A, B, C, D=None, Ts=1, x0=None):
        self.A = np.atleast_2d(A)
        self.B = np.atleast_2d(B)
        self.C = np.atleast_2d(C)
        
        if D is None:
            D = np.zeros((self.output_dim, self.input_dim))
        self.D = np.atleast_2d(D)

        self.Ts = Ts
        
        if x0 is None:
            x0 = np.zeros(self.state_dim)
        self.x0 = x0
        
        self.parametrize()
        
    @property
    def params(self):
        return {k:self[k][v] for k, v in self._idparams.items()}
    
    @params.setter
    def params(self, p):
        for k, v in self._idparams.items():
            self[k][v] = p[k]
        
    def parametrize(self, ption):
        self._idparams = {}
        
    @property
    def input_dim(self):
        return self.B.shape[1]
        
    @property
    def output_dim(self):
        return self.C.shape[0]
        
    @property
    def state_dim(self):
        return self.A.shape[0]
        
    @property
    def params_dim(self):
        return self.state_dim
        
    def as_tf(self):
        return self
        
    def sim(self, u, x0=None):
        #TODO: specialize for no inputs
        u = np.asarray(u)        
        nt = u.shape[0]
        
        x = np.zeros((nt, self.state_dim))
        if x0 is None:
            x[0,:] = self.x0
        else:
            x[0,:] = np.asarray(x0)
            
        A, B, C, D = self.A, self.B, self.C, self.D
        for i in xrange(nt-1):
            x[i+1,:] = np.dot(A, x[i,:]) + np.dot(B, u[i,:])
            
        y = np.dot(x, C.T) + np.dot(u, D.T)
        
        return y, x
        
    def dsim(self, u, x0=None):
        #TODO: specialize for no inputs
        u = np.asarray(u)        
        nt = u.shape[0]
        
        x = np.zeros((nt, self.state_dim, self.params_dim))
        if x0 is None:
            x[0,:] = self.x0
        else:
            x[0,:] = np.asarray(x0)
            
        A, B, C, D = self.A, self.B, self.C, self.D
        for i in xrange(nt-1):
            x[i+1,:] = np.dot(A, x[i,:]) + np.dot(B, u[i,:])
            
        y = np.dot(x, C.T) + np.dot(u, D.T)
        
        return y, x   

    
    
class DStochasticStateSpace(DStateSpace):
    """
    Stochastic discrete-time StateSpace System
    """
    def __init__(self, A, B, C, D=None, K=None, R=None, Ts=1, x0=None):
        pass
    
    def sim(self, u, compute_grad=False, stochastic=True):
        pass
    
if __name__ == "__main__":
    sys = DStateSpace([[.8,0],[.3,0]], [[1.],[2.]], [.5,.5])
    y, x = sys.sim(np.ones((100,1)))
    plt.plot(x,'.-')
    plt.plot(y,'.-')