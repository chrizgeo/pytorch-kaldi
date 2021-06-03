#!python3
import matplotlib.pyplot as plt
import numpy as np
import torch
import sru
import neural_networks
import sys
import os

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
            if 'weight' in innerKey.split("."):
                print(innerKey)
                numpy_array = modelDict[key][innerKey].cpu().detach().numpy()
                param_count += numpy_array.size
                print(numpy_array.size)
                #print(np.max(numpy_array))
                #print(np.min(numpy_array))
                zeros_count += np.count_nonzero(numpy_array == 0)
                print(np.count_nonzero(numpy_array == 0))
                #plt.hist(numpy_array, bins='auto')
                #plt.show()
compression = zeros_count/param_count
print("Compression ratio : %f " %compression)
