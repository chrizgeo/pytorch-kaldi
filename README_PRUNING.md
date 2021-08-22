# Pruning
The two files do the pruning implementation pruner.py and run_pruning_exp.py
pruner.py has the pruner object and the methods to iterate through the layers in both 
the LSTM and the SRU models. The Pruner class and the prune class method are used to implement this.

run_pruning_exp.py is a copy of run_exp.py. It has additions to read pruning related configs from 
the cfg files and init and call the pruner functions from pruner.py. It initialises the pruner object and
does pruning in the epochs which are specified in the cfg file.

# Iterative pruning
Iterative pruning is acheived by using different config files for each increase in the prune amount.
In the folder cfg/TIMIT_baselines, there are cfg files with names
TIMIT_LSTM_fbank_prune_**.cfg and TIMIT_SRU_fbank_prune_**.cfg
These config files does some epochs and trainig and training. 
The number of traning epochs,epochs in which pruning is done and the learning rate for each cfg file is empirically determined.

To make things easier, there are scripts to run the whole pruning experiments.
These scripts are named auto_exp_lstm.sh and auto_exp_sru.sh for LSTM and SRU respectively.
This scripts run the config files in sequence and the model parameters from the last run cfg file can be used for our further tests.