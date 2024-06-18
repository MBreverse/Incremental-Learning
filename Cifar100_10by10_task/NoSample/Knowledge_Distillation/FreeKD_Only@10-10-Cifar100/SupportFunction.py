from torchvision import  models
import torch.nn as nn
import torch

from ModelCollection import *

import os
import sys

from Config import *

cfg = Config()
record_root = cfg.record_root

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    #best_model = copy.deepcopy(model)
    torch.save(state, filename)
    
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
        model_dict = torch.load('{}/task{}_model.pth.tar'.format(record_root, task))
    else:
        model_dict = torch.load('{}/task{}_{}_model.pth.tar'.format(record_root, task, previous_epoch))
    cnn.load_state_dict(model_dict)
    
    #fixed layers except last layer
    # fixed_module_list = list(cnn.children())[:-1]

    # for m in fixed_module_list:
        # m.eval()
        
        # for name, value in m.named_parameters():
            # value.required_grad = False   

    return cnn

def Load_Kernel( insertion_check_list, kernel_sizes, task, training=False, previous_epoch = 0):
           
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


