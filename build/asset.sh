#!/bin/bash
#
#  Copyright (c) 2016 Intel Corporation 
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#


JAVA_PATH=$(find $BASE_DIR/$SOURCE_DIR/daaltk-core/ -name "daaltk-core*.zip")
PIP_PATH=$(find $BASE_DIR/$SOURCE_DIR/python/ -name "daaltk-*.tar.gz")
DAAL_INSTALL=$(find $BASE_DIR/$SOURCE_DIR/daal-install/ -name "daal-install")
LICENSES_PATH=$(find `pwd` -name "licenses*.zip")

echo java_path $JAVA_PATH
echo pip_path $PIP_PATH
echo daal-install $DAAL_INSTALL
echo license_path $LICENSES_PATH

echo $BASE_DIR/asset.sh daaltk-java $JAVA_PATH
$BASE_DIR/asset.sh daaltk-java $JAVA_PATH

echo $BASE_DIR/asset.sh daaltk-pip $PIP_PATH
$BASE_DIR/asset.sh daaltk-pip $PIP_PATH

echo $BASE_DIR/asset.sh daaltk-installer $DAAL_INSTALL
$BASE_DIR/asset.sh daaltk-installer $DAAL_INSTALL


#$BASE_DIR/asset.sh licenses $LICENSES_PATH
