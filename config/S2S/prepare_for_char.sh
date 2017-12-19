 python3 scripts/prepare-stc-data.py rawdata/DiaTest/huaweiFull2 message response data/DiaTestCharLevel  --no-tokenize --test-size 450000 --dev-size 450000 --vocab-size 30000 --shuffle --seed 1234 --verbose --shared-vocab --character-level
 # train a baseline model on this data
 nohup ./seq2seq.sh config/S2S/charlevel.yaml --train -v --debug >> charlog3.txt &

 nohup ./seq2seq.sh config/S2S/charlevel.yaml --train -v --decode >> charlog.txt


./seq2seq.sh config/S2S/charlevel.yaml -v --eval --gpu-id 1
./seq2seq.sh config/S2S/charlevel.yaml -v --decode --gpu-id 1