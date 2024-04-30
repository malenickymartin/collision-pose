#!/bin/bash

if [ $# -eq 0 ]; then
    echo "Usage: $0 <new_value>"
    exit 1
fi
mkdir -p data/datasets
cd data/datasets/${1}
wget https://bop.felk.cvut.cz/media/data/bop_datasets/${1}_test_bop19.zip
unzip ${1}_test_bop19.zip
rm ${1}_test_bop19.zip
mv test ${1}