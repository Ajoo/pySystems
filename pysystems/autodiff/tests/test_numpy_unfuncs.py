import sys
sys.path.append('C:\\Work\\pySystems\\pysystems')
import pytest

#from .. import numpy as dnp
import autodiff.numpy as dnp
import numpy as np

def make_hasattr_test(ufunc):
    attributes = ['nin', 'nout', 'nargs', 'ntypes', 'types', 'identity']    
    def test():
        for attribute in attributes:
            assert hasattr(ufunc, attribute)
    
    return test

test_add_hasattr = make_hasattr_test(dnp.add)

def make_unary_ufunc_test(ufunc):
    
    def test():
        a = dnp.darray(1+np.arange(9).reshape(3, 1, 3))
        
        c = ufunc(a)
        cf = ufunc.fd(a)
        
        dnp.testing.assert_allclose(c, cf, rtol=1e-5, verbose=False)
        
    return test

def make_binary_ufunc_test(ufunc):
    
    def test():
        a = dnp.darray(1+np.arange(9).reshape(3, 1, 3))
        b = dnp.darray(np.array([1., 5., 3.]))
        
        c = ufunc(a,b)
        cf = ufunc.fd(a,b)
        
        dnp.testing.assert_allclose(c, cf, rtol=1e-5, verbose=False)
        
    return test

ufuncs = dnp.ufuncs.__all__

for ufunc_name in ufuncs:
    ufunc = getattr(dnp, ufunc_name)
    if not isinstance(ufunc, dnp.DiffUFunc):
        continue
    if ufunc.nin == 1:
        globals()[u'test_' + ufunc_name] = make_unary_ufunc_test(ufunc)
    elif ufunc.nin == 2:
        globals()[u'test_' + ufunc_name] = make_binary_ufunc_test(ufunc)

if __name__ == '__main__':
    pytest.main()