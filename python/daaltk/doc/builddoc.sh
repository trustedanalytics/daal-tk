#!/usr/bin/env bash

# builds the python documentation using pdoc


NAME="[`basename $0`]"
DIR="$( cd "$( dirname "$0" )" && pwd )"
echo "$NAME DIR=$DIR"
cd $DIR

DAALTK_DIR="$(dirname "$DIR")"

tmp_dir=`mktemp -d`
echo $NAME created temp dir $tmp_dir

cp -r $DAALTK_DIR $tmp_dir
TMP_DAALTK_DIR=$tmp_dir/daaltk

# skip documenting the doc
rm -r $TMP_DAALTK_DIR/doc

echo $NAME pre-processing the python for the special doctest flags
python2.7 -m docutils -py=$TMP_DAALTK_DIR


echo $NAME cd $TMP_DAALTK_DIR
pushd $TMP_DAALTK_DIR > /dev/null

TMP_DAALTK_PARENT_DIR="$(dirname "$TMP_DAALTK_DIR")"
TEMPLATE_DIR=$DAALTK_DIR/doc/templates

# specify output folder:
HTML_DIR=$DAALTK_DIR/doc/html

# call pdoc
echo $NAME PYTHONPATH=$TMP_DAALTK_PARENT_DIR pdoc --only-pypath --html --html-dir=$HTML_DIR --template-dir $TEMPLATE_DIR --overwrite daaltk
PYTHONPATH=$TMP_DAALTK_PARENT_DIR pdoc --only-pypath --html --html-dir=$HTML_DIR --template-dir $TEMPLATE_DIR --overwrite daaltk

popd > /dev/null

# Post-processing:  Patch the "Up" links
echo $NAME post-processing the HTML
python2.7 -m docutils -html=$HTML_DIR

echo $NAME cleaning up...
echo $NAME rm $tmp_dir
rm -r $tmp_dir

echo $NAME Done.
