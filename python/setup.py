from setuptools import setup
from pip.req import parse_requirements
import os
import time


install_reqs = parse_requirements("requirements.txt", session=False)
reqs = [str(ir.req) for ir in install_reqs]

POST=os.getenv("DAALTK_POSTTAG","dev")
BUILD=os.getenv("DAALTK_BUILDNUMBER", "0")

VERSION=os.getenv("DAALTK_VERSION","0.7")

setup(
    # Application name:
    name="daaltk",

    version="{0}-{1}{2}".format(VERSION, POST, BUILD),

    # Application author details:
    author="trustedanalytics",


    # Packages
    packages=["daaltk"],

    # Include additional files into the package
    include_package_data=True,

    # Details
    url="https://github.com/trustedanalytics/daal-tk",

    #
    license="Apache 2.0",

    description="daal-tk is a library which provides an easy-to-use API for Python and Scala for using Intel DAAL models.",

    long_description=open("README.rst").read(),

    # Dependent packages (distributions)
    install_requires=reqs,

)
