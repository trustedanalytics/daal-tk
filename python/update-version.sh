#!/bin/bash

sed -i  "s|POST.*=.*|POST = \"$POST_TAG\"|g" setup.py
sed -i  "s|BUILD.*=.*|BUILD = \"$BUILD_NUMBER\"|g" setup.py
sed -i  "s|VERSION.*=.*|VERSION = \"$VERSION\"|g" setup.py
