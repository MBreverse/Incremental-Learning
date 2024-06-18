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
# Import datetime class from datetime module
from datetime import datetime


from ModelCollection import *
from TrainByFeature import *
from TestByFeature import *
from DatasetProcess import CIFAR_Split, Cifar_Feature_Split
from SupportFunction import *
from TestCnnByFeature import *
from Record import *
from Config import *

# returns current date and time
now = datetime.now()
print(now)

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
base_classes = cfg.base_classes
final_classes = cfg.total_classes
increm_classes = cfg.incre_classes
total_task = int( (final_classes - base_classes) / increm_classes) + 1 # 11(50/5)

data_root = cfg.data_root
branch = cfg.data_branch + '/feature_set'

# training  
task = 0
last_classes = base_classes
num_exemplar = cfg.num_exemplar #50


for i in range(total_task):
    # break
    
    task += 1
    
    dlabel=str(task)
    
    dataset_path = data_root + '/Cifar100_split'+ branch +'//split'+ dlabel +'_dataset.pth.tar'    
    
    dataset = torch.load(dataset_path)       
    
    print("task:",task)
    
    if task ==1:
        continue
        
        train_loader = DataLoader(dataset['train_true'], batch_size = 512, shuffle = True)
        base_model = CNN(base_classes)
        base_training(base_model, train_loader, epochs = 15)
        
        Write_Single_task_Training_Record(task = task , epochs = 15)
        
    else:
        if task <= 1:
            last_classes += increm_classes
            continue
        
        train_loader = DataLoader(dataset['train_true'], batch_size = 512, shuffle = True)        
        
        icl_training(task, last_classes, train_loader, epochs=100)
        
        Write_Single_task_Training_Record(task = task , epochs = 100)  
    
        last_classes += increm_classes
        # break
        # sys.exit()


#specify final task
final_task = total_task
# final_task = 4
# # testing 
# result_dict = test_all_task(base_classes = base_classes, increm_classes = increm_classes, final_task = final_task)
# # write test record
# for i in range(final_task):
    # Write_Single_Task_TestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)

result_dict = test_basic_cnn_for_all_training_task(base_classes = base_classes, increm_classes =increm_classes, final_classes = final_classes, final_task = final_task)
for i in range(final_task):
    Write_Single_Task_CNNTestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)

result_dict = test_cnn_all_task_prec(base_classes = base_classes, increm_classes =increm_classes, final_task = final_task)
for i in range(final_task):
    Write_Single_Task_TestPrec_Record(task_prec_list = result_dict[str(i+1)], train_task = i+1)

# result_dict= test_sscnn_for_all_training_task(base_classes = base_classes, increm_classes = increm_classes,  final_classes = final_classes, final_task = final_task)
# for i in range(final_task):
    # Write_Single_Task_SSCNNTestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)    

# result_dict = test_all_task_prec(base_classes = base_classes, increm_classes =increm_classes, final_task = final_task)
# for i in range(final_task):
    # Write_Single_Task_TestPrec_Record(task_prec_list = result_dict[str(i+1)], train_task = i+1) 
