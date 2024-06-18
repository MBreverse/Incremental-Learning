from torchvision import  models
import torch.nn as nn
import torch

import os
import sys

from ModelCollection import *
from Config import *

device = "cuda" if torch.cuda.is_available() else "cpu"
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
    
def one_hot(labels, class_size):

    targets = torch.zeros(labels.size(0), class_size)
    
    for i, label in enumerate(labels):
        targets[i, label] = 1
    return targets.to(device)    


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
    
class WeightKeeper():

    def __init__(self, inital_model, interval=[0,50]):
        
        #load last fc layer old weight
        layer_names = []
        for param_tensor in inital_model.state_dict():
            layer_names.append(param_tensor)

        #read last fc layer weight and bias
        last_weight = inital_model.state_dict()[layer_names[len(layer_names) - 2 ]]   #最後一層(weight)
        last_bias = inital_model.state_dict()[layer_names[len(layer_names) - 1 ]]    # 最後一層(bias層)

        start, end = interval
        previous_weight = last_weight[start:end].detach().clone()
        previous_bias = last_bias[start:end].detach().clone()   
        
        self.origin_weight = previous_weight
        self.origin_bias = previous_bias   
        self.interval = interval
    
    def recover(self, updated_model):

        #cover fc3 weight and bias
        layer_names = []
        for param_tensor in updated_model.state_dict():
            layer_names.append(param_tensor)
            
        state_dict = updated_model.state_dict()

        last_weight = state_dict[layer_names[len(layer_names) - 2 ]]   #最後一層(weight)
        last_bias = state_dict[layer_names[len(layer_names) - 1 ]]    # 最後一層(bias層)            
        
        start, end = self.interval
        last_weight[start:end] = self.origin_weight 
        last_bias[start:end] = self.origin_bias

        state_dict[layer_names[len(layer_names) - 2 ]] = last_weight
        state_dict[layer_names[len(layer_names) - 1 ]] = last_bias

        updated_model.load_state_dict(state_dict)
        return updated_model
  
class LossRatioFinetune(): 
    
    def __init__(self):
        
        self.init_beta = cfg.init_ratio
        self.cur_beta = self.init_beta
        self.inc_rate = cfg.incre_ratio
        self.half_cycle_epoch = cfg.half_cycle
        self.counter = 0
    
    def update_beta(self, epoch):
           
        if self.counter < self.half_cycle_epoch:
        
            self.cur_beta += self.inc_rate

        elif self.counter >= self.half_cycle_epoch and self.counter < (self.half_cycle_epoch*2):
        
            self.cur_beta = self.cur_beta
        
        elif self.counter == (self.half_cycle_epoch*2):
            
            self.cur_beta = self.cur_beta
            self.counter = 0

        self.counter += 1     
    
    def get_beta(self):
        return self.cur_beta

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
    

if __name__ == "__main__":
    
    print(__name__)
