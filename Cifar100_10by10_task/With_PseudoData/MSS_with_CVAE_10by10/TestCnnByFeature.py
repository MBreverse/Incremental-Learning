import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import vgg16, resnet50

from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

import time
import copy
import sys
import os
import os.path
import json

from ModelCollection import *
from DatasetProcess import CIFAR_Split,Cifar_Feature_Split
from SupportFunction import *
from Config import *


device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()


# class split hyper-parameter
base_classes = cfg.base_classes
increm_classes = cfg.incre_classes

#end of training classes
final_classes = cfg.final_classes
final_task = cfg.test_tasks

#specified testing hyper-parameter
previous_epoch = 0   
batch_size = cfg.batch_size
feature_dim = cfg.feature_dim

#kernel hyper-parameter
insertion_check_list = cfg.insert_check_list
kernel_size = cfg.ss_sizes


#dataset path
root = cfg.data_root
dataset_name = cfg.data_name
branch = cfg.data_branch

def test_basic_cnn_for_all_training_task(final_classes = 50, final_task = 1):
    
    task_acc_list_dict = {}
    for cur_task in range(1, final_task+1):
        
        task_acc_list_dict.update({str(cur_task) : []})
    
        print("loading model...")
        cur_classes = base_classes + increm_classes * (cur_task-1)
        cnn = Load_Model(cur_classes, cur_task, previous_epoch = previous_epoch)

        model = cnn.to(device)
        model.eval()


        root = './data'
        
        for test_task in range(1, cur_task+1):
            
            dlabel = str(test_task)

            dataset_path = root + dataset_name + branch + '//split'+dlabel+'_dataset.pth.tar'    # 新任務的資料路徑
            
            dataset = torch.load(dataset_path)
            
            test_loader = DataLoader(dataset['val_true'], batch_size = 256, shuffle = False)
                
            cur_classes = base_classes + increm_classes * (test_task-1)
            
            print("test task:{} / classes:{}".format(test_task, cur_classes))
            with torch.no_grad():

                acc_sample = 0
                total_sample = 0
                total_loss = 0
                
                for i, (feature, label) in enumerate(test_loader):
                    print('.', end = '')
                    
                    feature = feature.view(-1,2048)
                    feature = feature.to(device)
                    label = label.to(device)                    
                             
                    output = model(feature,mode="f")
                 
                    final_pred = torch.argmax(output, dim = 1)
                    final_pred = final_pred.view(label.shape)
                    acc_sample += torch.sum(final_pred == label).item()
                    total_sample += label.shape[0]
                    
            print("\ntest accuracy:",acc_sample/total_sample)
                
            task_acc_list_dict[str(cur_task)].append(acc_sample/total_sample)   
        
    return task_acc_list_dict


def test_sscnn_for_all_training_task(final_classes = 50, final_task = 1):
    
    insertion_check_list = [1,1,1]
    kernel_size
    
    
    task_acc_list_dict = {}
    for cur_task in range(1, final_task+1):
        
        task_acc_list_dict.update({str(cur_task) : []})
    
        print("loading model...")
        cur_classes = base_classes + increm_classes * (cur_task-1)
        cnn = Load_Model(cur_classes, cur_task, previous_epoch = 0)
        
        print("loading kernel...")
        all_kernel_list = []
        # load all kernel
        for i in range(2, cur_task + 1):
            
            kernel_list1 = Load_Kernel( insertion_check_list, kernel_size, i, training=False, previous_epoch = 0)
                
            all_kernel_list.append(kernel_list1)

        #conbine model and kernel
        model = SS_Model( cnn, insertion_check_list, kernel_size, all_kernel_list )

        model = model.to(device)

        model.eval()        

  
        for test_task in range(1, cur_task+1):
            
            dlabel = str(test_task)
            
            dataset_path = root + dataset_name + branch + '//split'+dlabel+'_dataset.pth.tar'     # 新任務的資料路徑
            
            dataset = torch.load(dataset_path)
            
            test_loader = DataLoader(dataset['val_true'], batch_size = 256, shuffle = False)
                
            cur_classes = base_classes + increm_classes * (test_task-1)
            
            print("test task:{} / classes:{}".format(test_task, cur_classes))
            with torch.no_grad():

                acc_sample = 0
                total_sample = 0
                total_loss = 0
                
                for i, (feature, label) in enumerate(test_loader):
                    print('.', end = '')
                    
                    feature = feature.view(-1,2048)
                    feature = feature.to(device)
                    label = label.to(device)                    
                             
                    output = model(feature, inp_mode="f")
                 
                    final_pred = torch.argmax(output[-1], dim = 1)
                    final_pred = final_pred.view(label.shape)
                    acc_sample += torch.sum(final_pred == label).item()
                    total_sample += label.shape[0]
                    
            print("\ntest accuracy:",acc_sample/total_sample)
                
            task_acc_list_dict[str(cur_task)].append(acc_sample/total_sample)   
        
    return task_acc_list_dict
