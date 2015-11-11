# -*- coding: utf-8 -*-

import numpy as np
from collections import defaultdict
import functools

#Alternative DiffNDArray implementation as a np.ndarray subclass
class DiffNDArray(np.ndarray):
    def __init__(self, value, d=None):
        if d is not None:        
            self.d = d
            #self.track()
        else:
            self.d = {}
            self.track()
        
        #derivative last
        self.data = np.concatenate((np.asarray(value)[...,np.newaxis],), axis=-1)
        
        #derivative first        
        #self.data = np.vstack((np.asarray(value)[np.newaxis],))
        
    @property
    def value(self):
        return self.data[...,0]
        
    @property
    def derivative(self, wrt):
        try:
            return self.data[...,self.dindex[wrt]]
        except KeyError:
            return np.zeros(self.shape + wrt.shape)
    
    @property
    def shape(self):
        return super(DiffNDArray, self).shape[:-2]