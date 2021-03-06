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


WD=target/


if [ "$POST_TAG" == "rc" ]; then
    MODULE="daaltk-core-$VERSION$POST_TAG$BUILD_NUMBER"
else
    MODULE="daaltk-core-$VERSION.$POST_TAG$BUILD_NUMBER"
fi

pushd $WD

	mkdir -p $MODULE/dependencies

	cp daaltk-core*.jar $MODULE/

	for source in `find \`pwd\` -iname "*sources.jar"`
	do
	if [ "$source" != "" ]; then
	echo remove source file $source
	rm  $source
	fi
	done

	cp -Rv dependencies/* $MODULE/dependencies/

	pushd $MODULE
	ln -s dependencies lib
	popd

	zip --symlinks -r $MODULE.zip $MODULE

	rm -rf $MODULE

popd
