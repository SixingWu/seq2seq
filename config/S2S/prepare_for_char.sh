  python3 scripts/prepare-stc-data.py rawdata/DiaTest/huaweiFull2 message response data/CharLevel2  --no-tokenize --test-size 100000 --dev-size 100000 --vocab-size 30000 --shuffle --seed 1234 --verbose --shared-vocab --character-level

 # train a baseline model on this data
 nohup ./seq2seq.sh config/S2S/charlevel.yaml --train -v --debug --beam-size 5 >> charlog5.txt &

 nohup ./seq2seq.sh config/S2S/charlevel.yaml -v --decode >> charlog.txt


./seq2seq.sh config/S2S/charlevel.yaml -v --eval --gpu-id 1
./seq2seq.sh config/S2S/charlevel.yaml -v --decode --gpu-id 1


nohup python3 scripts/prepare-stc-data.py rawdata/DiaTest/huaweiFull2 message response data/SubLevel2  --no-tokenize --test-size 100000 --dev-size 100000 --vocab-size 30000 --shuffle --seed 1234 --verbose --shared-vocab --subwords  >> log.txt &
 # train a baseline model on this data

  nohup ./seq2seq.sh config/S2S/subwords.yaml --train -v --gpu-id 1 --debug --beam-size 5 >> subwords.txt &