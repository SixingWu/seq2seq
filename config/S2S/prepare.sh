#!/usr/bin/env bash

# 测试数据

raw_data=raw_data/DiaTest
data_dir=data/DisTest

rm -rf ${data_dir}
mkdir -p ${data_dir}

prepare-stc-data.py rawdata/DiaTest/huawei message response data/DiaTest  --no-tokenize --test-size 10000 --dev-size 10000 --vocab-size 30000 --shuffle --seed 1234 --verbose --shared-vocab
