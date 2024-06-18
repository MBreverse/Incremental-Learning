from torchvision import  models
import torch.nn as nn
import torch

import os
import sys

from ModelCollection import *
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
        model_dict = torch.load(record_root + '/task{}_{}.pth.tar'.format(task, cfg.save_cnn_name))

    else:
        model_dict = torch.load(record_root + '/task{}_{}_{}.pth.tar'.format(task, previous_epoch, cfg.save_cnn_name))
    cnn.load_state_dict(model_dict)
    
    # #fixed layers except last layer
    # fixed_module_list = list(cnn.children())[:-1]

    return cnn

def Load_Kernel( insertion_check_list, kernel_sizes, task, training=False, previous_epoch = 0):
           
    # load kernel
    if previous_epoch == 0:
        kernel_path =  record_root + '/task{}_{}.pth.tar'.format(task, cfg.save_ss_name)
    else:
        kernel_path = record_root + '/task{}_{}_{}.pth.tar'.format(task, previous_epoch, cfg.save_ss_name)
        
    if os.path.exists(kernel_path):
            
        kernel_list1= Multi_Scale_Shift_Kernel(insertion_check_list, kernel_sizes)
        
        kernel_dict = torch.load(kernel_path)
        kernel_list1.load_state_dict(kernel_dict)
        
    else:
        kernel_list1 = None
    
    
    return kernel_list1

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

class Incre_Class_Iter():
    
    def __init__(self, base_class_size, increm_class_size, total_class_size):
    
        self.num_base_class = base_class_size
        self.num_increm_class = increm_class_size
        self.num_total_class = total_class_size
        self.total_task = (total_class_size - base_class_size)/ increm_class_size + 1

    def get_cur_class_digits(self, task):
        
        if task <= 0 :
            raise "task number is negative!!"
        
        if task == 1:
            end_class_digit = self.num_base_class - 1
            start_class_digit = 0
        else:
            end_class_digit = (task - 1) * self.num_increm_class + self.num_base_class - 1
            start_class_digit = end_class_digit - self.num_increm_class + 1
            
        return start_class_digit, end_class_digit
        
class SSInitializer():

    def __init__(self, scale_idxs, shift_idxs):
        
        self.fix_scale_idxs = scale_idxs
        self.fix_shift_idxs = shift_idxs
        
    def recover(self, ss_set):

        #fixed the part kernel weight or bias
        for i, kernel in enumerate(ss_set.kernel_list):
        
            if i in self.fix_scale_idxs:
            
                if kernel != None:
                    kernel.shift.weight.data.fill_(0.)
                    
            if i in self.fix_shift_idxs:
            
                if kernel != None:
                    kernel.scale.weight.data.fill_(1.)
        
        return ss_set


if __name__ == "__name__":
    
    print(__name__)


