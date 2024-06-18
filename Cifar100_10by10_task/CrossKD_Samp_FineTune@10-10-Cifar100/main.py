import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import vgg16, resnet50


import os
import sys
import os.path
import json
import math


from ModelCollection import *
from TrainByFeature import *
from TestByFeature import *
from DatasetProcess import CIFAR_Split, Cifar_Feature_Split
from SupportFunction import *
from TestCnnByFeature import *
from Record import *
from Config import *

# Import datetime class from datetime module
from datetime import datetime
# returns current date and time
now = datetime.now()
print(now)

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()

base_classes = cfg.base_classes
final_classes = cfg.final_classes
increm_classes = cfg.incre_classes

# task classes iterator along incremental task
# ClassIter = Incre_Class_Iter(base_classes, increm_classes, final_classes)

total_task = int( (final_classes - base_classes) / increm_classes) + 1 # 11(50/5)

# data relative parameter
data_root = cfg.data_root
branch = cfg.data_branch
code_root = cfg.code_root
data_name = cfg.data_name

# training  
task = 0
last_classes = base_classes
# num_exemplar = cfg.num_exemplar
# fixed_memory = 2000
num_exemplar = cfg.num_exemplar


total_task = int( (final_classes - base_classes) / increm_classes) + 1 # 11(50/5)

for i in range(total_task):
    # break
    
    task += 1
    
    dlabel=str(task)
    
    dataset_path = data_root + data_name + branch +'//split'+ dlabel +'_dataset.pth.tar'    
    
    dataset = torch.load(dataset_path)       
    
    print("task:",task)
    
    
    # total_digit = base_classes
    if task ==1:
        # continue
        
        train_loader = DataLoader(dataset['train_true'], batch_size = cfg.batch_size, shuffle = True)
        base_model = Recoverable_CNN(base_classes)
        base_training(base_model, train_loader)
        
        jsonfile = record_root + "/task1_{}_history.json".format(cfg.base_cnn_train_his_name)
        Write_SingleTaskTrain_Record(task = task, json_path = jsonfile, item_names = ['loss', 'acc'])
        

        # # for sampling data from old task    
        # num_exemplar = int(math.floor(fixed_memory/ base_classes))
        
        exemplar_set = Exemplar_Feature_Set(dataset = dataset['train_true'], num_exemplar = num_exemplar, label_format = "origin")
        torch.save(exemplar_set,code_root+'/exemplar_set.pth.tar')
        # sys.exit()
        
    else:
        if task <= 1:
            last_classes += increm_classes
            continue

        old_exemplar_set = torch.load(code_root+'/exemplar_set.pth.tar')
        new_dataset = dataset['train_true']
        
        # ConcateSet = torch.utils.data.ConcatDataset([old_exemplar_set, new_dataset])        
        # train_loader = DataLoader(ConcateSet, batch_size = cfg.batch_size, shuffle = True)        
        # train_loader = DataLoader(dataset['train_true'], batch_size = cfg.batch_size, shuffle = True)        
        
        icl_training(task, old_exemplar_set, new_dataset)
        
        jsonfile = record_root + "/task{}_{}_history.json".format(task, cfg.icl_cnn_train_his_name)
        Write_SingleTaskTrain_Record(task = task, json_path = jsonfile, item_names = ['loss', 'acc', 'loss1', 'loss2'])
        
        
        # num_exemplar = int(math.floor(fixed_memory/ (last_classes + increm_classes)))
        
        print("Exemplar from new dataset:")
        new_exemplar_set = Exemplar_Feature_Set(dataset = dataset['train_true'], num_exemplar = num_exemplar, label_format = "origin")
        
        print("combine new/old dataset:")
        ConcateSet = torch.utils.data.ConcatDataset([old_exemplar_set, new_exemplar_set])   
        torch.save(ConcateSet,code_root+'/exemplar_set.pth.tar')   
        
        last_classes += increm_classes
        # break

# assert False


#specify final task
final_task = total_task

result_dict = test_basic_cnn_for_all_training_task(base_classes = base_classes, increm_classes =increm_classes, final_classes = final_classes, final_task = final_task)
for i in range(final_task):
    Write_Single_Task_CNNTestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)


assert False  

result_dict = test_all_task_acc(base_classes = base_classes, increm_classes = increm_classes, final_task =final_task)
# write test record
for i in range(final_task):
    Write_Single_Task_TestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)

assert False  
  
# testing 
result_dict = test_all_task(base_classes = base_classes, increm_classes = increm_classes, final_task = final_task)
# write test record
for i in range(final_task):
    Write_Single_Task_TestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)

result_dict = test_basic_cnn_for_all_training_task(base_classes = base_classes, increm_classes =increm_classes, final_classes = final_classes, final_task = final_task)
for i in range(final_task):
    Write_Single_Task_CNNTestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)

result_dict= test_sscnn_for_all_training_task(base_classes = base_classes, increm_classes = increm_classes,  final_classes = final_classes, final_task = final_task)
for i in range(final_task):
    Write_Single_Task_SSCNNTestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)    

result_dict = test_all_task_prec(base_classes = base_classes, increm_classes =increm_classes, final_task = final_task)
for i in range(final_task):
    Write_Single_Task_TestPrec_Record(task_prec_list = result_dict[str(i+1)], train_task = i+1) 
