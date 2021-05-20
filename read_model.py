#!python3
import matplotlib.pyplot as plt
import numpy as np
import torch
import sru
import neural_networks

modelFile = open("exp/TIMIT_SRU_fbank_simple_prune/exp_files/final_architecture1.pkl", 'rb')
modelDict = torch.load(modelFile)
for key in modelDict:
    print(key)
    if key == 'model_par':
        for innerKey in modelDict[key]:
            if 'weight' in innerKey.split("."):
                print(innerKey)
                numpy_array = modelDict[key][innerKey].cpu().detach().numpy()
                print(numpy_array.size)
                print(np.max(numpy_array))
                print(np.min(numpy_array))
                print(np.count_nonzero(numpy_array == 0))
                #plt.hist(numpy_array, bins='auto')
                #plt.show()
