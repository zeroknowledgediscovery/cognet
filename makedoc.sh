#!/bin/bash

pdoc --html cognet/ -o docs/ -c latex_math=True -f --template-dir docs/dark_templates

cp -r docs/cognet/* docs
