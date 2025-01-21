#!/bin/sh -e
set -x

autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place "sft" --exclude=__init__.py
isort "sft"
black "sft" -l 80
