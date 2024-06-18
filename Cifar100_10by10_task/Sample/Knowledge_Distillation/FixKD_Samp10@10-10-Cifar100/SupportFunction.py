from torchvision import  models
import torch.nn as nn
import torch

from ModelCollection import *
from Config import *

import os
import sys

    
cfg = Config()
record_root = cfg.record_root
previous_epoch = 0
    
def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)

def module_reconstruction(module, origin_classes, target_classes, first = False):    
    
    print("Reconstruct last layer...")   
        
    layer_names = []   
    for param_name in module.state_dict():
        layer_names.append(param_name)        
      
    
    last_weight = module.state_dict()[layer_names[len(layer_names) - 2 ]]   #最後一層(weight)
    last_bias = module.state_dict()[layer_names[len(layer_names) - 1 ]]    # 最後一層(bias層)
    
    # load weight and bias from fc3 
    #create the n+1 classe including 1 old class and n new classes
    if first == False:
        # last module
        inp_size = module.weight.shape[1]
        f = nn.Linear(inp_size, target_classes) 
        f.weight.data[0:origin_classes] = last_weight
        f.bias.data[0:origin_classes] = last_bias          
        module = f

    else:
        # first module
        out_size = module.weight.shape[0]
        f = nn.Linear(target_classes,out_size)        
        f.weight.data[:,0:origin_classes] = last_weight
        f.bias.data[:] = last_bias  
        module = f    
    
    return module

def model_reconstruction(model, origin_classes, target_classes):    
    
    print("Reconstruct last layer...")
    layer_names = []
    for param_tensor in model.fc2.state_dict():
        layer_names.append(param_tensor)        
      
    
    last_weight = model.fc2.state_dict()[layer_names[len(layer_names) - 2 ]]   #最後一層(weight)
    last_bias = model.fc2.state_dict()[layer_names[len(layer_names) - 1 ]]    # 最後一層(bias層)
    
    # load weight and bias from fc3 
    #create the n+1 classe including 1 old class and n new classes
    f = nn.Linear(1024, target_classes) 

    
    f.weight.data[0:origin_classes] = last_weight
    f.bias.data[0:origin_classes] = last_bias
  
    model.fc2 = f
    
    print("complete!")
    return model

def Load_Model(model_classes, task, previous_epoch = 0):
    
    #load model
    cnn = CNN(model_classes)
    
    if previous_epoch == 0:
        model_dict = torch.load(record_root + '/task{}_{}.pth.tar'.format(task, cfg.save_cnn_name))
    else:
        model_dict = torch.load(record_root + '/task{}_{}_{}.pth.tar'.format(task, previous_epoch, cfg.save_cnn_name))
    cnn.load_state_dict(model_dict)
    
    #fixed layers except last layer
    # fixed_module_list = list(cnn.children())[:-1]

    return cnn

def Load_Kernel( insertion_check_list, kernel_sizes, task, training=False, previous_epoch = 0):

    cfg = Config()
    record_root = cfg.record_root
    
    # load kernel
    if previous_epoch == 0:
        kernel_path = '{}/task{}_kernel.pth.tar'.format(record_root, task)
    else:
        kernel_path = '{}/task{}_{}_kernel.pth.tar'.format(record_root, task, previous_epoch)
        
    if os.path.exists(kernel_path):
            
        kernel_list1= Multi_Scale_Shift_Kernel(insertion_check_list, kernel_sizes)
        
        kernel_dict = torch.load(kernel_path)
        kernel_list1.load_state_dict(kernel_dict)
        
    else:
        kernel_list1 = None
        
    return kernel_list1
    
def Load_All_Task_SS(insertion_check_list, ss_sizes, cur_task, training = False):
    
    all_ss_set_list = []
    # load all kernel
    for i in range(2, cur_task + 1):

        ss_set = Load_Kernel( insertion_check_list, ss_sizes, i, training=False, previous_epoch = previous_epoch)
        
        all_ss_set_list.append(ss_set)
    
    return all_ss_set_list

def Load_EN_Kernel(insertion_check_list, kernel_sizes, task, training=False, previous_epoch = 0):

    cfg = Config()
    record_root = cfg.record_root
    
    # load kernel
    if previous_epoch == 0:
        kernel_path = '{}/task{}_en_kernel.pth.tar'.format(record_root, task)
    else:
        kernel_path = '{}/task{}_{}_en_kernel.pth.tar'.format(record_root, task, previous_epoch)
        
    if os.path.exists(kernel_path):
            
        kernel_list1= Multi_Scale_Shift_Kernel(insertion_check_list, kernel_sizes)
        
        kernel_dict = torch.load(kernel_path)
        kernel_list1.load_state_dict(kernel_dict)
        
    else:
        kernel_list1 = None
    
    
    return kernel_list1

def Load_DE_Kernel(insertion_check_list, kernel_sizes, task, training=False, previous_epoch = 0):

    cfg = Config()
    record_root = cfg.record_root
    
    # load kernel
    if previous_epoch == 0:
        kernel_path = '{}/task{}_de_kernel.pth.tar'.format(record_root, task)
    else:
        kernel_path = '{}/task{}_{}_de_kernel.pth.tar'.format(record_root, task, previous_epoch)
        
    if os.path.exists(kernel_path):
            
        kernel_list1= Multi_Scale_Shift_Kernel(insertion_check_list, kernel_sizes)
        
        kernel_dict = torch.load(kernel_path)
        kernel_list1.load_state_dict(kernel_dict)
        
    else:
        kernel_list1 = None
    
    
    return kernel_list1    


