#
# /*
# // Copyright (c) 2016 Intel Corporation
# //
# // Licensed under the Apache License, Version 2.0 (the "License");
# // you may not use this file except in compliance with the License.
# // You may obtain a copy of the License at
# //
# //      http://www.apache.org/licenses/LICENSE-2.0
# //
# // Unless required by applicable law or agreed to in writing, software
# // distributed under the License is distributed on an "AS IS" BASIS,
# // WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# // See the License for the specific language governing permissions and
# // limitations under the License.
# */
#

from setuptools import setup
from pip.req import parse_requirements
import os
import time


install_reqs = parse_requirements("requirements.txt", session=False)
reqs = [str(ir.req) for ir in install_reqs]

POST="DEV"
BUILD="0"
VERSION="0.7"

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
