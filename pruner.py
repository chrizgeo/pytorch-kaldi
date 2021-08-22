import os
import sys
import configparser
import torch
import random
import numpy as np
import setuptools
from distutils.util import strtobool
import torch.nn as nn
import torch.nn.utils.prune as prune
import neural_networks
from data_io import read_lab_fea
from utils import model_init
import pickle
import copy

""" The pruner class for pruning.
    This is used as a container to store pruning methods and objects.
    This class creates a neural network for the passed architecture and also contains metadata related to pruning. """ 
class Pruner:
    # init
    def __init__(self, cfg_file, pt_file, prune_method='lnstructured', n=1, layers_to_prune=[1], prune_amounts = {1:0.2}, prune_amounts_proj = {1:0.2}):
        print("Pruner -INFO- Init prune object")
        #prune method, lnstructured and unstructured 
        self.prune_method=prune_method
        #amounts to prune the layers. If there are more than one layer to be pruned,
        #the amount is a value from 0.0 to 1.0 (eg. 0.7 prunes 70% of the weights)
        self.prune_amount = prune_amounts
        #used if the projection layers(sru) need to be pruned to different amounts
        self.prune_amount_proj = prune_amounts_proj
        #store the prune masks for this object.
        self.prune_mask = {}
        #n values for ln methods. default 1, not using any other values
        self.n = n
        #layers to prune, read from the config file.
        self.layers_to_prune = layers_to_prune
        #read pruning related info from the config file
        if not (os.path.exists(cfg_file)):
            sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
            sys.exit(0)
        else:
            self.config = configparser.ConfigParser()
            self.config.read(cfg_file)
        self.out_folder = self.config["exp"]["out_folder"]
        #store mask tensor as a pkl file
        self.mask_file = self.out_folder + "prune_mask.pkl"
        #load model weights from file
        self.arch_dict = torch.load(pt_file)
        #platform related config
        self.use_cuda = strtobool(self.config["exp"]["use_cuda"])
        self.multi_gpu = strtobool(self.config["exp"]["multi_gpu"])
        self.model = self.config["model"]["model"].split("\n")
        #model related config
        self.pt_file = pt_file
        self.to_do = self.config["exp"]["to_do"] 
        print("Pruner - INFO - init complete")
         

    # called from the run_pruning_exp function. This does one round of pruning
    #TODO keep the pruning history somewhere
    def prune(self, cfg_file, pt_file):
        print("Pruner -INFO- Pruning")
        #print(pt_file)
        self.pt_file = pt_file
        self.arch_dict = torch.load(pt_file)
        
        # taken from core.py to init the model we want to prune
        shared_list = []
        read_lab_fea(cfg_file, True, shared_list, self.out_folder)
        fea_dict = shared_list[2]
        arch_dict = shared_list[4]
        # the cost is not used here, but kept for syntax
        [nns, cost] = model_init(fea_dict, self.model, self.config, arch_dict, self.use_cuda, self.multi_gpu, self.to_do)
        for net in nns.keys():
            if self.multi_gpu:
                nns[net] = torch.nn.DataParallel(nns[net])

        # get the name of the arch1 network layer
        self.net = list(nns.keys())[0].split("_")[0]
        if self.multi_gpu:  
            self.net_module = nns[list(nns.keys())[0]].module
        else:
            self.net_module = nns[list(nns.keys())[0]]

        #pickle.dump(self.prune_mask, open(self.mask_file, "wb"))

        print("Pruner-INFO-loading state dict")
        self.net_module.load_state_dict(self.arch_dict["model_par"])
        if self.net == "SRU":
            self._sru_prune_once()
            self.arch_dict["model_par"] = self.net_module.state_dict()    
        elif self.net == "LSTM":
            self._lstm_prune_once()
            self.arch_dict["model_par"] = self.net_module.state_dict()
        #torch.save(self.arch_dict_orig, pt_file)

    def _sru_prune_once(self):
        # init pruning object
        # the hooks and the object needs to be created every time the pruning is done 
        # since the training is spread out with each epoch creating a model dict
        if self.net == "SRU":
            if self.prune_method == 'lnstructured':
                self.prune_obj_weight = prune.LnStructured(0.2, n=self.n)
                self.prune_obj_weight._tensor_name = 'weight'
                self.prune_obj_weight_proj = prune.LnStructured(0.2, n=self.n)
                self.prune_obj_weight_proj._tensor_name = 'weight_proj'
            elif self.prune_method == 'unstructured':
                self.prune_obj_weight = prune.L1Unstructured(0.2)
                self.prune_obj_weight._tensor_name = 'weight'
                self.prune_obj_weight_proj = prune.L1Unstructured(0.2)
                self.prune_obj_weight_proj._tensor_name = 'weight_proj'
                #self._sru_init_pruner()        
        
        # init the pruning hooks.
        # it is not really necessary to have these hooks in the current workflow, 
        # but i couldnt find a better way to do the pruning without 
        # these hooks and the weight_mask buffers
        for module in self.net_module.children():
            for _sub_module_list in module.children():
                for i in self.layers_to_prune:
                    if self.prune_method == 'lnstructured':
                        self.prune_obj_weight.apply(_sub_module_list[i], name='weight', amount=self.prune_amount[i], n=self.n, dim=1)
                        self.prune_obj_weight_proj.apply(_sub_module_list[i], name='weight_proj', amount=self.prune_amount_proj[i], n=self.n, dim=1)
                    elif self.prune_method == 'unstructured':
                        self.prune_obj_weight.apply(_sub_module_list[i], name='weight', amount=self.prune_amount[i])
                        self.prune_obj_weight_proj.apply(_sub_module_list[i], name='weight_proj', amount=self.prune_amount_proj[i])

        # prune the individual weight and weight_proj tensors
        for module in self.net_module.children():
            for _sub_module_list in module.children():
                for i in self.layers_to_prune:
                # for i in range(1,4):
                    _sub_module_list[i].weight = self.prune_obj_weight.prune(_sub_module_list[i].weight, default_mask=_sub_module_list[i].weight_mask )
                    _sub_module_list[i].weight_proj = self.prune_obj_weight_proj.prune(_sub_module_list[i].weight_proj, default_mask=_sub_module_list[i].weight_proj_mask)
        
        # remove hooks
        # removing the hooks so the model parameters we pass to 
        # training will have the same names
        for module in self.net_module.children():
            for _sub_module_list in module.children():
                for i in self.layers_to_prune:
                # for i in range(1,4):
                    self.prune_obj_weight.remove(_sub_module_list[i])
                    self.prune_obj_weight_proj.remove(_sub_module_list[i])
    
        # save the weights to the file
        # the model parameters are saved in the file withe the same name we started with, 
        # so effectively we have a pruned version which will be used by the next epoch of training. 
        self.arch_dict["model_par"] = self.net_module.state_dict()
        torch.save(self.arch_dict, self.pt_file)
        # TODO save the prune mask to file - this part was removed on modification of code do adapt LSTM.
        #pickle.dump(self.prune_mask, open(self.mask_file, "wb"))

    #internal function used for development and debugging 
    """ def _print_parameters_sru(self):
        print("print parameters")
        for module in self.net_module.children():
            for _sub_module_list in module.children():
                #print(_sub_module_list[1])
                for i in self.layers_to_prune:
                # for i in range(1,4):
                    print("Pruner version")
                    print("sru.rnn_lst.%d.weight" %i)
                    print(_sub_module_list[i].weight)
                    #print("sru.rnn_lst.%d.weight_mask" %i)
                    #print(_sub_module_list[i].weight_mask)
                    print("Original version")
                    key_name = "sru.rnn_lst.{}.weight".format(i)
                    print(key_name)
                    print(self.arch_dict_orig["model_par"][key_name]) """

    """ Prune once the lstm network
        The same method as with SRU is repeated here, with the only difference with how the 
        layers are iterated"""
    def _lstm_prune_once(self):
        # init pruning object
        # the hooks and the object needs to be created every time the pruning is done 
        # since the training is spread out with each epoch creating a model dict
        if self.prune_method == 'lnstructured':
                self.prune_obj_weight = prune.LnStructured(0.2, n=self.n)
                self.prune_obj_weight._tensor_name = 'weight'
        elif self.prune_method == 'unstructured':
                self.prune_obj_weight = prune.L1Unstructured(0.2)
                self.prune_obj_weight._tensor_name = 'weight'

        # init the pruning hooks.
        # it is not really necessary to have these hooks in the current workflow, 
        # but i couldnt find a better way to do the pruning without 
        # these hooks and the weight_mask buffers
        for _module_list in self.net_module.children():
            for i in self.layers_to_prune:
                #print(_module_list[i])
                if _module_list[i] == nn.Linear:
                    if self.prune_method == 'lnstructured':
                        self.prune_obj_weight.apply(_module_list[i], name='weight', amount=self.prune_amount[i], n=self.n, dim=1)
                    elif self.prune_method == 'unstructured':
                        self.prune_obj_weight.apply(_module_list[i], name='weight', amount=self.prune_amount[i])
        
        # prune the individual weights
        for _module_list in self.net_module.children():
            for i in self.layers_to_prune:
                #print(_module_list[i])
                if _module_list[i] == nn.Linear:
                    _module_list[i].weight = self.prune_obj_weight.prune(_module_list[i].weight, default_mask=_module_list[i].weight_mask )

        # remove hooks
        # removing the hooks so the model parameters we pass to 
        # training will have the same names
        for _module_list in self.net_module.children():
            for i in self.layers_to_prune:
                #print(_module_list[i])
                if _module_list[i] == nn.Linear:
                    self.prune_obj_weight.remove(_module_list[i])
        
        # save the weights to the file
        # the model parameters are saved in the file withe the same name we started with, 
        # so effectively we have a pruned version which will be used by the next epoch of training. 
        self.arch_dict["model_par"] = self.net_module.state_dict()
        torch.save(self.arch_dict, self.pt_file)
        # TODO save the prune mask to file