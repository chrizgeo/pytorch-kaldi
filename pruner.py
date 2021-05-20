import os
import sys
import configparser
#for pruning
import torch
import random
import numpy as np
from distutils.util import strtobool
import torch.nn.utils.prune as prune
import neural_networks
from data_io import read_lab_fea
from utils import model_init

#use the l1 unstructured pruner from inbuilt library
l1_pruner = prune.L1Unstructured(.2)
l1_pruner._tensor_name = 'weight'

#args - cfg_file for the last processed chunk, pretrained file to get optimiser parameters
def pruner_main(cfg_file, pt_file, out_folder, method=None):

    arch1_dict = torch.load(pt_file)   
    # This function processes the current chunk using the information in cfg_file. In parallel, the next chunk is load into the CPU memory
    print(cfg_file)
    # Reading chunk-specific cfg file (first argument-mandatory file)
    if not (os.path.exists(cfg_file)):
        sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
        sys.exit(0)
    else:
        config = configparser.ConfigParser()
        config.read(cfg_file)

    # Setting torch seed
    seed = int(config["exp"]["seed"])
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    to_do = config["exp"]["to_do"]
    info_file = config["exp"]["out_info"]
    # Reading config parameters
    output_folder = config["exp"]["out_folder"]
    use_cuda = strtobool(config["exp"]["use_cuda"])
    multi_gpu = strtobool(config["exp"]["multi_gpu"])
    model = config["model"]["model"].split("\n")

    shared_list = []
    read_lab_fea(cfg_file, True, shared_list, out_folder)
    data_name = shared_list[0]
    data_end_index = shared_list[1]
    fea_dict = shared_list[2]
    lab_dict = shared_list[3]
    arch_dict = shared_list[4]
    data_set = shared_list[5]

    [nns, costs] = model_init(fea_dict, model, config, arch_dict, use_cuda, multi_gpu, to_do)
    for net in nns.keys():
        if multi_gpu:
            nns[net] = torch.nn.DataParallel(nns[net])
    
    #get the name of the arch1 network layer
    arch1_net = list(nns.keys())[0]
    simple_pruner_sru(nns[arch1_net])
    #arch1_dict["model_par"] = nns["SRU_layers"].state_dict
    if multi_gpu:
        arch1_dict["model_par"] = nns[arch1_net].module.state_dict()
    else:
        arch1_dict["model_par"] = nns[arch1_net].state_dict()
    torch.save(arch1_dict, pt_file)

def simple_pruner_sru(arch_dict):
    for module in arch_dict.module.children():
        #print("nns[SRU_layers].children()")
        #print(module)
        #the four rnn layers are in the module which is a child of the SRU module - This is SRU implementation specific
        for _sub_module_list in module.children():
            #we skip the layer 0 and start from layer 1
            #layer 0 does not contribute much to the size of the model
            for i in range (1,4):
                #print(_sub_module_list[i].weight.shape)
                l1_pruner.apply(_sub_module_list[i], name='weight', amount=0.2)                
                l1_pruner.prune(_sub_module_list[i].weight)
                l1_pruner.remove(_sub_module_list[i])
                print(_sub_module_list[i].weight)
    
    return True

def unwrap_model(model):
	# loops through all layers of model, including inside modulelists and sequentials
	layers = []
	def unwrap_inside(modules):
		for m in modules.children():
			if isinstance(m,nn.Sequential):
				unwrap_inside(m)
			elif isinstance(m,nn.ModuleList):
				for m2 in m:
					unwrap_inside(m2)
			else:
				layers.append(m)
	unwrap_inside(model)
	return nn.ModuleList(layers)

def myPruner(model,inp):
    return True