#!/bin/bash

COMMENT=" updated "


a=$((`awk -F= '{print $2}' version.py  | sed "s/\s*'//g" | awk -F. '{print $NF}'` + 1))
VERSION=`awk -F= '{print $2}' version.py  | sed "s/\s*'//g" | awk -F. 'BEGIN{OFS="."}{print $1,$2}'`.$a


echo __version__ = \'$VERSION\'
#exit

echo __version__ = \'$VERSION\' > version.py

../gitscripts_/gitscript.sh $COMMENT

git add * -v
git commit -m $COMMENT
git push origin main
git tag $VERSION -m $COMMENT
git push --tags

rm -rf dist

python3 setup.py sdist
python3 setup.py bdist_wheel
twine check dist/*
twine upload dist/*


