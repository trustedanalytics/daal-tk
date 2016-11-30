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


sed -i  "s|DAAL_VERSION.*=.*|DAAL_VERSION = \"$DAAL_VERSION\"|g" main.go
sed -i  "s|DAALTK_VERSION.*=.*|DAALTK_VERSION = \"$DAALTK_VERSION\"|g" main.go
sed -i  "s|SPARKTK_VERSION.*=.*|SPARKTK_VERSION = \"$SPARKTK_VERSION\"|g" main.go
