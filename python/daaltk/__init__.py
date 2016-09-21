import sys
import os
from daaltk.daal import Daal
from sparktk.tkcontext import TkContext

LIB_DIR="dependencies"
DAALTK_HOME_ENV_VAR = "DAALTK_HOME"

def get_main_object(tc):
    """ Returns the library's main object (the Daal class), which is used to access DAAL models and operations."""
    return Daal(tc)

def get_library_dirs():
    """Returns the folders which contain all the jars required to run daaltk"""
    if DAALTK_HOME_ENV_VAR not in os.environ:
        raise Exception("Required environment variable %s not set" % DAALTK_HOME_ENV_VAR)

    daaltk_home = os.environ[DAALTK_HOME_ENV_VAR]
    return [daaltk_home, os.path.join(daaltk_home, LIB_DIR)]

def get_loaders(tc):
    """ Called by spark-tk to get daal-tk loaders. """
    if not isinstance(tc, TkContext):
        raise RuntimeError("tc parameter must be a TkContext, but recieved %s." % type(tc))
    return tc.sc._jvm.org.trustedanalytics.daaltk.saveload.Loaders.getLoaders()
