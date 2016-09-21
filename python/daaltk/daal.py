from sparktk.lazyloader import get_lazy_loader
import os
from sparktk import TkContext

class Daal(object):
    def __init__(self, tc=None):
        if not (isinstance(tc, TkContext)):
            raise TypeError("DAAL init requires a valid TkContext.  Received type: %s." % type(tc))
        self._tc = tc

        self._parent_path = os.path.dirname(os.path.abspath(__file__))
        self._package_name = "daaltk"

    @property
    def models(self):
        """Access to the various models of daaltk"""
        return get_lazy_loader(self, "models", parent_path=self._parent_path, package_name=self._package_name, implicit_kwargs={'tc': self})

    @property
    def operations(self):
        """Access to the various operations of daaltk"""
        return get_lazy_loader(self, "operations", parent_path=self._parent_path, package_name=self._package_name, implicit_kwargs={'tc': self}).operations