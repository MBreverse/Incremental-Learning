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

# task classes iterator along incremental task
ClassIter = Incre_Class_Iter(base_classes, increm_classes, final_classes)

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
        
        # train_loader = DataLoader(dataset['train_true'], batch_size = cfg.batch_size, shuffle = True)
        base_training(dataset['train_true'])
        
        jsonfile = record_root + "/task1_{}_history.json".format(cfg.base_cnn_train_his_name)
        Write_SingleTaskTrain_Record(task = task, json_path = jsonfile, item_names = ['loss', 'acc'])
        
        jsonfile = record_root + '/task1_{}_history.json'.format(cfg.base_cvae_train_his_name)
        Write_SingleTaskTrain_Record(task = task, json_path = jsonfile, item_names = ['loss', 'loss1', 'loss2'])        
        
    else:
        if task <= 1:
            last_classes += increm_classes
            continue

        # generate sudo exemplar for old_task data
        PseudoSetList = []
        for past_task in range(1, task):
            start_class_digit, end_class_digit = ClassIter.get_cur_class_digits(past_task) 
            PseudoSetList.append(CVAE_PseudoExempar( sample_per_class = num_exemplar, class_digits = [start_class_digit, end_class_digit] , task = task - 1 ))  

        '''Combine exemplar and new dataset'''
        new_dataset = dataset['train_true']    
        ConcateSet = torch.utils.data.ConcatDataset(PseudoSetList + [new_dataset])    

        icl_training(task, ConcateSet)
        
        jsonfile = record_root + "/task{}_{}_history.json".format(task, cfg.icl_cnn_train_his_name)
        Write_SingleTaskTrain_Record(task = task, json_path = jsonfile, item_names = ['loss', 'acc'] )
        
        jsonfile = record_root + "/task{}_{}_history.json".format(task, cfg.icl_cvae_train_his_name)
        Write_SingleTaskTrain_Record(task = task, json_path = jsonfile, item_names = ['loss', 'loss1', 'loss2'])          

        last_classes += increm_classes
        # break
# sys.exit()


#specify final task
final_task = total_task

# testing 
# result_dict = test_all_task_recall_on_HierachyPred( final_task = final_task)
# for i in range(final_task):
    # Write_Single_Task_TestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1, total_task = total_task)

result_dict = test_basic_cnn_for_all_training_task(final_classes = final_classes, final_task = final_task)
for i in range(final_task):
    Write_Single_Task_CNNTestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)

result_dict = test_BasicCNN_Prec(final_classes = final_classes, final_task = final_task)
for i in range(final_task):
    Write_Single_Task_TestPrec_Record(task_prec_list = result_dict[str(i+1)], train_task = i+1, total_task = final_task)    

# result_dict= test_sscnn_for_all_training_task(final_task = final_task)
# for i in range(final_task):
    # Write_Single_Task_SSCNNTestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)    

# result_dict = test_all_task_with_all_SSModel(final_task = final_task, mode = "prec")
# for i in range(final_task):
    # Write_Single_Task_TestPrec_Record(task_prec_list = result_dict[str(i+1)], train_task = i+1, total_task = final_task) 
