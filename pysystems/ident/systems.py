from .. import DStateSpace

def extension_method(cls):
    def decorator(func):
        setattr(cls, func.__name__, func)
        return func
    return decorator
    
def extension_class(name, bases, namespace):
    assert len(bases) == 1, "Exactly one base class required"
    base = bases[0]
    for name, value in namespace.iteritems():
        if name != "__metaclass__":
            setattr(base, name, value)
    return base


class IdentDStateSpace(DStateSpace):
    __metaclass__ = extension_class
    
    def fit(self, u, y):
        pass