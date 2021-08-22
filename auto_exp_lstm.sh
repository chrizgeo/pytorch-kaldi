#!/bin/bash

#Run the pruning scripts to get a finalised pruned and retrained model
#which GPU to use since one will be full sometimes due to other users on the server using it
CUDA_VISIBLE_DEVICES=0

rm -r /data/home/chrgeo19/exp/TIMIT_LSTM_fbank_prune_01
rm -r /data/home/chrgeo19/exp/TIMIT_LSTM_fbank_prune_02
rm -r /data/home/chrgeo19/exp/TIMIT_LSTM_fbank_prune_03
rm -r /data/home/chrgeo19/exp/TIMIT_LSTM_fbank_prune_04
rm -r /data/home/chrgeo19/exp/TIMIT_LSTM_fbank_prune_05
rm -r /data/home/chrgeo19/exp/TIMIT_LSTM_fbank_prune_06
rm -r /data/home/chrgeo19/exp/TIMIT_LSTM_fbank_prune_07
rm -r /data/home/chrgeo19/exp/TIMIT_LSTM_fbank_prune_08

#first roound of pruning
python3 run_pruning_exp.py cfg/TIMIT_baselines/TIMIT_LSTM_fbank_prune_01.cfg

#second round
python3 run_pruning_exp.py cfg/TIMIT_baselines/TIMIT_LSTM_fbank_prune_02.cfg

#third round
python3 run_pruning_exp.py cfg/TIMIT_baselines/TIMIT_LSTM_fbank_prune_03.cfg

#fourth round
#python3 run_pruning_exp.py cfg/TIMIT_baselines/TIMIT_LSTM_fbank_prune_04.cfg

#fifth round
#python3 run_pruning_exp.py cfg/TIMIT_baselines/TIMIT_LSTM_fbank_prune_05.cfg

#sixth round
#python3 run_pruning_exp.py cfg/TIMIT_baselines/TIMIT_LSTM_fbank_prune_06.cfg

#seventh round
#python3 run_pruning_exp.py cfg/TIMIT_baselines/TIMIT_LSTM_fbank_prune_07.cfg

#eighth round
#python3 run_pruning_exp.py cfg/TIMIT_baselines/TIMIT_LSTM_fbank_prune_08.cfg