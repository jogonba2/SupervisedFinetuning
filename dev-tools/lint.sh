#!/usr/bin/env bash

set -e
set -x

mypy "sft"
flake8 "sft" --ignore=E501,W503,E203,E402,E704
black "sft" --check -l 80
