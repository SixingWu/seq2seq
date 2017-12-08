#!/usr/bin/env bash

# 测试数据

raw_data=raw_data/DiaTest
data_dir=data/DisTest

rm -rf ${data_dir}
mkdir -p ${data_dir}

scripts/prepare-data.py ${raw_data}/DiaTest post response ${data_dir} --no-tokenize \
--dev-corpus ${raw_data}/train \
--test-corpus ${raw_data}/test \
--vocab-size 30000 --shuffle --seed 1234
