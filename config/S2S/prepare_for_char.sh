 python3 scripts/prepare-stc-data.py rawdata/DiaTest/huaweiFull2 message response data/DiaTestCharLevel  --no-tokenize --test-size 100000 --dev-size 100000 --vocab-size 50000 --shuffle --seed 1234 --verbose --shared-vocab --character-level
 # train a baseline model on this data
 nohup ./seq2seq.sh config/S2S/charlevel.yaml --train -v --debug >> charlog2.txt &

 nohup ./seq2seq.sh config/S2S/charlevel.yaml --train -v --decode >> charlog.txt
