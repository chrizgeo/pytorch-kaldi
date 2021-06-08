#!python3
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import sru
import neural_networks
import sys
import os
import math

pickle_file = sys.argv[1]
if not os.path.exists(pickle_file):
    sys.stderr.write("ERROR: The config file %s does not exist!\n" % (pickle_file))
    sys.exit(0)
else:
    modelFile = open(pickle_file, 'rb')

modelDict = torch.load(modelFile)
param_count = 0
zeros_count = 0
for key in modelDict:
    print(key)
    if key == 'model_par':
        for innerKey in modelDict[key]:
            _param_type = innerKey.split(".")[3]
            print(_param_type)
            if _param_type == 'weight' or _param_type == 'weight_proj':
                print(innerKey)
                np_weights = modelDict[key][innerKey].cpu().detach().numpy()
                param_count += np_weights.size
                print(np_weights.size)
                np_weights_flat = np_weights.flatten()
                series_data = pd.Series(np_weights_flat)
                series_data.plot.hist(grid=True, bins=40, rwidth=0.95, color='#0504aa')
                plt.title(innerKey)
                plt.xlabel('Weight')
                plt.ylabel('Count')
                plt.grid(axis='y', alpha=0.75)
                plt.show()
                #hist, bin_edges = np.histogram (np_weights, bins=10, range = None, normed = None, weights = None, density = None)
                zeros_count += np.count_nonzero(np_weights == 0)
                print(np.count_nonzero(np_weights == 0))
            else:
                np_weights = modelDict[key][innerKey].cpu().detach().numpy()
                param_count += np_weights.size

compression = zeros_count/param_count
print("Compression ratio : %f " %compression)
