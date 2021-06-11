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

#the pruner class 
class Pruner:
    def __init__(self, cfg_file, pt_file, prune_method='lnstructured', n=1, layers_to_prune=[1], epochs_to_prune=[0], prune_amounts = {1:0.2}, prune_amounts_proj = {1:0.2}):
        print("Init prune objects")
        print(epochs_to_prune)
        self.prune_method=prune_method
        self.prune_amount = {}
        self.prune_amount_proj = {}
        num_pruning_epochs = len(epochs_to_prune)
        self.current_pruning_epoch=1
        #iteratively increase the prune amount with each step, starting with 0.05 (5%)
        self.prune_amount_step={}
        self.prune_amount_step_proj={}
        for layer in layers_to_prune:
            max_prune_amount = prune_amounts[layer]
            max_prune_amount_proj = prune_amounts_proj[layer]
            self.prune_amount_step[layer] = (max_prune_amount-0.05) /num_pruning_epochs
            self.prune_amount_step_proj[layer] = (max_prune_amount_proj - 0.05) /num_pruning_epochs

        self.prune_mask = {}
        self.n = n
        self.layers_to_prune = layers_to_prune
        if not (os.path.exists(cfg_file)):
            sys.stderr.write("ERROR: The config file %s does not exist!\n" % (cfg_file))
            sys.exit(0)
        else:
            config = configparser.ConfigParser()
            config.read(cfg_file)
        out_folder = config["exp"]["out_folder"]
        self.mask_file = out_folder + "prune_mask.pkl"
        self.arch_dict_orig = torch.load(pt_file)
        self.arch_dict  = copy.deepcopy(self.arch_dict_orig)
        #self.arch_dict = torch.load(pt_file)
        #self._init_dict()
        self.use_cuda = strtobool(config["exp"]["use_cuda"])
        self.multi_gpu = strtobool(config["exp"]["multi_gpu"])
        self.model = config["model"]["model"].split("\n")
        self.pt_file = pt_file
        to_do = config["exp"]["to_do"]
        shared_list = []
        read_lab_fea(cfg_file, True, shared_list, out_folder)
        data_name = shared_list[0]
        data_end_index = shared_list[1]
        fea_dict = shared_list[2]
        lab_dict = shared_list[3]
        arch_dict = shared_list[4]
        data_set = shared_list[5]

        [nns, costs] = model_init(fea_dict, self.model, config, arch_dict, self.use_cuda, self.multi_gpu, to_do)
        for net in nns.keys():
            if self.multi_gpu:
                nns[net] = torch.nn.DataParallel(nns[net])

        #get the name of the arch1 network layer
        self.net = list(nns.keys())[0].split("_")[0]
        if self.multi_gpu:  
            self.net_module = nns[list(nns.keys())[0]].module
        else:
            self.net_module = nns[list(nns.keys())[0]]

        pickle.dump(self.prune_mask, open(self.mask_file, "wb"))

        print("loading state dict")
        #self.net_module.load_state_dict(self.arch_dict_orig["model_par"])
        self.net_module.load_state_dict(self.arch_dict["model_par"])

        #init the pruning objects 
        print("init")
        if self.net == "SRU":
            self._create_default_mask_sru()
            if self.prune_method == 'lnstructured':
                self.prune_obj_weight = prune.LnStructured(0.05, n=self.n)
                self.prune_obj_weight._tensor_name = 'weight'
                self.prune_obj_weight_proj = prune.LnStructured(0.05, n=self.n)
                self.prune_obj_weight_proj._tensor_name = 'weight_proj'
                #self._structured_sru_init()
            elif self.prune_method == 'unstructured':
                self.prune_obj_weight = prune.L1Unstructured(0.05, n=self.n)
                self.prune_obj_weight._tensor_name = 'weight'
                self.prune_obj_weight_proj = prune.L1Unstructured(0.05, n=self.n)
                self.prune_obj_weight_proj._tensor_name = 'weight_proj'
        elif self.net == "LSTM":
            self._create_default_mask_lstm()


    def _create_default_mask_sru(self):
        print("create_default_mask_sru")
        self.prune_mask['w'] = {}
        self.prune_mask['wp'] = {}
        for module in self.net_module.children():
            for _sub_module_list in module.children():
                for i in self.layers_to_prune:
                #for i in range(1,4):
                    default_mask = torch.ones(_sub_module_list[i].weight.size())
                    #print(_sub_module_list[i].weight)
                    if _sub_module_list[i].weight.is_cuda:
                        default_mask = default_mask.cuda()
                    self.prune_mask['w'][i] = default_mask
                    #print(_sub_module_list[i].weight_proj)
                    default_mask = torch.ones(_sub_module_list[i].weight_proj.size())
                    if _sub_module_list[i].weight_proj.is_cuda:
                        default_mask = default_mask.cuda()
                    self.prune_mask['wp'][i] = default_mask
        pickle.dump(self.prune_mask, open(self.mask_file, "wb"))
        print("create_default_mask_sru : Exit")
 
    def _create_default_mask_lstm(self):
        print("create_default_mask_lstm")

    #call this from the run_pruning_exp function. This does one round of pruning
    #TODO keep the pruning history somewhere so that we can have some history for our next step
    def prune(self, cfg_file, pt_file):
        print("Pruning %d \n" % self.current_pruning_epoch)
        print(pt_file)
        self.arch_dict_orig = torch.load(pt_file)
        if self.net == "SRU":
            #self._copy_parameters_sru()
            self.net_module.load_state_dict(self.arch_dict["model_par"])
            #self._print_parameters_sru()
            if self.prune_method == 'lnstructured':
                #self._structured_sru_init
                self._structured_sru_prune()
            elif self.prune_method == 'unstructured':
                self._unstructured_sru()
            #self.arch_dict["model_par"] = self.net_module.state_dict()
            #self._update_parameters_sru()
            #self._structured_sru_final()    
        elif self.net == "LSTM":
            if self.prune_method == 'lnstructured':
                self.strucrured_lstm()
            elif self.prune_method == 'unstructured':
                self.unstructured_lstm()
        self.current_pruning_epoch+=1
        self._print_parameters_sru()
        torch.save(self.arch_dict_orig, pt_file)

    #call this to save the pruned arch_dict in the final model
    def finalise_pruning(self, cfg_file, pt_file):
        self._structured_sru_final(pt_file)
        #self._print_parameters_sru

    def _print_parameters_sru(self):
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
                    #print("Original version")
                    #key_name = "sru.rnn_lst.{}.weight".format(i)
                    #print(key_name)
                    #print(self.arch_dict_orig["model_par"][key_name])


    def _copy_parameters_sru(self):
        for i in range(1,4):
            key_name = "sru.rnn_lst.{}.weight".format(i)
            key_name_orig = "sru.rnn_lst.{}.weight_orig".format(i)
            key_name_mask = "sru.rnn_lst.{}.weight_mask".format(i)
            #self.arch_dict["model_par"][key_name] = self.arch_dict_orig["model_par"][key_name]
            self.arch_dict["model_par"][key_name_orig] = self.arch_dict_orig["model_par"][key_name]
            key_name = "sru.rnn_lst.{}.weight_proj".format(i)
            key_name_orig = "sru.rnn_lst.{}.weight_proj_orig".format(i)
            key_name_mask = "sru.rnn_lst.{}.weight_proj_mask".format(i)
            #self.arch_dict["model_par"][key_name] = self.arch_dict_orig["model_par"][key_name]
            self.arch_dict["model_par"][key_name_orig] = self.arch_dict_orig["model_par"][key_name]

    def _update_parameters_sru(self):
        self.arch_dict["model_par"] = self.net_module.state_dict()
        print("Pruner obj")
        for key in self.arch_dict["model_par"].keys():
            print(key)
        print("original")
        for key in self.arch_dict_orig["model_par"].keys():
            print(key)
        for module in self.net_module.children():
            for _sub_module_list in module.children():
                for i in self.layers_to_prune:
                # for i in range(1,4):
                    key_name = "sru.rnn_lst.{}.weight".format(i)
                    #key_name_orig = "sru.rnn_lst.{}.weight_orig".format(i)
                    self.arch_dict_orig["model_par"][key_name] = _sub_module_list[i].weight
                    key_name = "sru.rnn_lst.{}.weight_proj".format(i)
                    #key_name_orig = "sru.rnn_lst.{}.weight_proj_orig".format(i)
                    self.arch_dict_orig["model_par"][key_name] = _sub_module_list[i].weight_proj
        torch.save(self.arch_dict_orig, self.pt_file)

    #internal function for applying the hooks and copying the values
    def _structured_sru_init(self):
        for module in self.net_module.children():
            for _sub_module_list in module.children():
                for i in self.layers_to_prune:
                # for i in range(1,4):
                    self.prune_obj_weight.apply(_sub_module_list[i], name='weight', amount=0.05, n=self.n, dim=1)
                    self.prune_obj_weight_proj.apply(_sub_module_list[i], name='weight_proj', amount=0.05, n=self.n, dim=1)
        #self._print_parameters_sru() 
        print("Model's state_dict:")
        for param_tensor in self.net_module.state_dict():
            print(param_tensor, "\t", self.net_module.state_dict()[param_tensor].size())
            self.arch_dict["model_par"] = self.net_module.state_dict()
        #self._update_parameters_sru()

    def _structured_sru_prune(self):
        for module in self.net_module.children():
            for _sub_module_list in module.children():
                for i in self.layers_to_prune:
                # for i in range(1,4):
                    print("Pruning amount")
                    print(self.prune_amount_step[i])
                    self.prune_obj_weight.apply(_sub_module_list[i], name='weight', amount=self.prune_amount_step[i], n=self.n, dim=1)
                    self.prune_obj_weight_proj.apply(_sub_module_list[i], name='weight_proj', amount=self.prune_amount_step[i], n=self.n, dim=1)
                    #self.prune_obj_weight.compute_mask(_sub_module_list[i].weight_orig, self.prune_mask['w'][i])
                    #_sub_module_list[i].weight = self.prune_obj_weight.apply_mask(_sub_module_list[i].weight_orig)
                    #self.prune_obj_weight.compute_mask(_sub_module_list[i].weight_proj_orig, self.prune_mask['wp'][i])
                    #_sub_module_list[i].weight_proj = self.prune_obj_weight_proj.apply_mask(_sub_module_list[i].weight_proj_orig)
                    #self.prune_obj_weight.prune(_sub_module_list[i].weight)
                    #self.prune_obj_weight_proj.prune(_sub_module_list[i].weight_proj)   

        print("After pruning")
        #self._print_parameters_sru()
        self._update_parameters_sru()
        pickle.dump(self.prune_mask, open(self.mask_file, "wb"))

    def _structured_sru_final(self):
        print("removing hooks")
        for module in self.net_module.children():
            for _sub_module_list in module.children():
                for i in self.layers_to_prune:
                # for i in range(1,4):
                    self.prune_obj_weight.remove(_sub_module_list[i])
                    self.prune_obj_weight_proj.remove(_sub_module_list[i])
        self.arch_dict["model_par"] = self.net_module.state_dict()
        torch.save(self.arch_dict, self.    pt_file)
 
    def _unstructured_sru(self):
        for module in self.net_module.children():
            for _sub_module_list in module.children():
                for i in self.layers_to_prune:
                # for i in range(1,4):
                    self.prune_obj_weight._tensor_name = 'weight'
                    self.prune_obj_weight.apply(_sub_module_list[i], name='weight', amount=self.prune_amount[i])
                    self.prune_mask['w'][i-1] = self.prune_obj_weight.compute_mask(_sub_module_list[i].weight, self.prune_mask['w'][i-1])
                    self.prune_obj_weight.prune(_sub_module_list[i].weight,self.prune_mask['w'][i-1])
                    self.prune_obj_weight.remove(_sub_module_list[i])
                    self.prune_obj_weight_proj._tensor_name = 'weight_proj'
                    self.prune_obj_weight_proj.apply(_sub_module_list[i], name='weight_proj', amount=self.prune_amount_proj[i])
                    self.prune_mask['wp'][i-1] = self.prune_obj_weight.compute_mask(_sub_module_list[i].weight_proj, self.prune_mask['wp'][i-1])
                    self.prune_obj_weight_proj.prune(_sub_module_list[i].weight, self.prune_mask['wp'][i-1])
                    self.prune_obj_weight_proj.remove(_sub_module_list[i])

        pickle.dump(self.prune_mask, open(self.mask_file, "wb"))

    def structured_lstm(self):
        pass
    def unstructured_lstm(self):
        pass