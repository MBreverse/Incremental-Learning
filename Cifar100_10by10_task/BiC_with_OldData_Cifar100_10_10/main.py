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
branch = cfg.data_branch
dataset_name = cfg.dataset_name

# training  
task = 0
last_classes = base_classes
num_exemplar = cfg.num_exemplar #50

for i in range(total_task):
    # break
    task += 1
    
    dlabel=str(task)
    
    dataset_path = data_root + dataset_name + branch +'//split'+ dlabel +'_dataset.pth.tar'    
    
    dataset = torch.load(dataset_path)       
    
    print("task:",task)
    
    if task ==1:
        # continue
        
        train_loader = DataLoader(dataset['train_true'], batch_size = 512, shuffle = True)
        base_model = CNN(base_classes)
        base_training(base_model, train_loader)
        
        jsonfile = record_root + "/task1_{}_history.json".format(cfg.base_cnn_train_his_name)
        Write_SingleTaskTrain_Record(task = task, json_path = jsonfile, item_names = ['loss', 'acc'])
        
        # # for sampling data from old task      
        exemplar_set = Exemplar_Feature_Set(dataset = dataset['train_true'], num_exemplar = num_exemplar, label_format = "origin")
        torch.save(exemplar_set, cfg.code_root + '/exemplar_set.pth.tar')
        # sys.exit()
        
    else:
        if task <= 1:
            last_classes += increm_classes
            continue
        
        '''Combine exemplar and new dataset'''
        print("Combine Exemplar and new data:")
        old_exemplar_set = torch.load( cfg.code_root + '/exemplar_set.pth.tar')
        new_dataset = dataset['train_true']
               
        old_set_len = len(old_exemplar_set)
        new_set_len = len(new_dataset)
        total_len = old_set_len +  new_set_len   

        old_val_len = int(total_len * 0.1 * 0.5)# balance
        new_val_len = int(total_len * 0.1 * 0.5)# balance
        
        old_train_len = old_set_len - old_val_len
        new_train_len = new_set_len - new_val_len
 
        # split dataset into val / train set for BiC and icl respectively
        
        # old_set_len = len(old_exemplar_set)
        # old_train_len = int(0.9 * old_set_len)
        # old_val_len = old_set_len - int(0.9 * old_set_len)
        
        old_train, old_valid = torch.utils.data.random_split(old_exemplar_set, [old_train_len, old_val_len])

        # new_set_len = len(new_dataset)
        # new_train_len = int(0.9 * new_set_len)
        # new_val_len = new_set_len - int(0.9 * new_set_len)

        new_train, new_valid = torch.utils.data.random_split(new_dataset, [new_train_len, new_val_len])
        
        ConcateSet = torch.utils.data.ConcatDataset([old_train, new_train])
        # sys.exit()
        
        # '''use it if need to train alternative classifier with new classes + 1 single class'''
        # next_class = last_classes + increm_classes
        # new_dataset = Shift_Label_Dataset(new_dataset, origin_class_logit_range = [last_classes, next_class-1], target_class_logit_range = [1,5]) 
        
        ConcateSet = torch.utils.data.ConcatDataset([old_exemplar_set, new_dataset])
        
        train_loader = DataLoader(ConcateSet, batch_size = 512, shuffle = True)
        # train_loader = DataLoader(dataset['train_true'], batch_size = 512, shuffle = True)        
        icl_training(task, last_classes, train_loader)
        
        jsonfile = record_root + "/task{}_{}_history.json".format(task, cfg.icl_cnn_train_his_name)
        Write_SingleTaskTrain_Record(task = task, json_path = jsonfile, item_names = ['loss', 'acc', "loss1", "loss2"] )
        
        # train for BiC process
        ConcateSet = torch.utils.data.ConcatDataset([old_valid, new_valid])
        train_loader = DataLoader(ConcateSet, batch_size = 512, shuffle = True)
        BiC_training(task, train_loader)
        
        jsonfile = record_root + "/task{}_{}_history.json".format(task, cfg.bic_model_train_his_name)
        Write_SingleTaskTrain_Record(task = task, json_path = jsonfile, item_names = ['loss', 'acc'] )        
        
        '''prepare data for next task training'''
        ''' using exemplar set only previous task'''
        print("Exemplar from new dataset:")
        new_exemplar_set = Exemplar_Feature_Set(dataset = dataset['train_true'], num_exemplar = num_exemplar, label_format = "origin")
        
        print("combine new/old dataset:")
        ConcateSet = torch.utils.data.ConcatDataset([old_exemplar_set, new_exemplar_set])   
        torch.save(ConcateSet, cfg.code_root + '/exemplar_set.pth.tar')    
    
        
        last_classes += increm_classes
        # break
        # sys.exit()


#specify final task
final_task = total_task

result_dict = test_bic_cnn_for_all_training_task( final_classes = final_classes , final_task = final_task)

for i in range(final_task):
    Write_Single_Task_TestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1, total_task = total_task)

# testing 
# result_dict = test_all_task_recall_on_HierachyPred( final_task = final_task)
# for i in range(final_task):
    # Write_Single_Task_TestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1, total_task = total_task)

# result_dict = test_basic_cnn_for_all_training_task(final_classes = final_classes, final_task = final_task)
# for i in range(final_task):
    # Write_Single_Task_CNNTestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)

# result_dict = test_BasicCNN_Prec(final_classes = final_classes, final_task = final_task)
# for i in range(final_task):
    # Write_Single_Task_TestPrec_Record(task_prec_list = result_dict[str(i+1)], train_task = i+1, total_task = final_task)    

# result_dict= test_sscnn_for_all_training_task(final_task = final_task)
# for i in range(final_task):
    # Write_Single_Task_SSCNNTestAcc_Record(task_acc_list = result_dict[str(i+1)], train_task = i+1)    

# result_dict = test_all_task_with_all_SSModel(final_task = final_task, mode = "prec")
# for i in range(final_task):
    # Write_Single_Task_TestPrec_Record(task_prec_list = result_dict[str(i+1)], train_task = i+1, total_task = final_task) 
