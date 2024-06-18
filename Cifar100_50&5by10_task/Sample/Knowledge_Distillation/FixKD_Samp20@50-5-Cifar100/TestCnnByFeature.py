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


#dataset path
cfg = Config()

root = cfg.data_root
dataset_name = cfg.dataset_name
branch = cfg.data_branch + '/feature_set'

increm_classes = cfg.incre_classes
base_classes = cfg.base_classes

#specified testing hyper-parameter
previous_epoch = 0   
batch_size = 128
feature_dim =2048
kernel_size = cfg.ss_sizes



class TaskPrecision():
    
    def __init__(self, task):
    
        self.total_pred_task = task
        self.EachTaskTP = {}
        self.EachTaskFP = {}
        
        for t in range(1,task+1):
            self.EachTaskTP.update({str(t):0})
            self.EachTaskFP.update({str(t):{}})
        
            for other_task in range(1,task+1):
                if t != other_task:
                    self.EachTaskFP[str(t)].update({str(other_task):0})

    def ConutTP(self, pred_task, count):
        
        self.EachTaskTP[str(pred_task)] += count

    def CountFP(self, true_task, pred_task, count):
    
        self.EachTaskFP[str(true_task)][str(pred_task)] += count

    def ComputeTaskPrecision(self):
        
        task_prec_list = []
        
        for true_task in range(1, self.total_pred_task+1):
            
            sum_FP = 0            
            for pred_task in range(1, self.total_pred_task+1): 
                
                if true_task != pred_task:
                    sum_FP += self.EachTaskFP[str(pred_task)][str(true_task)]

            if self.EachTaskTP[str(true_task)] != 0:
                task_prec = self.EachTaskTP[str(true_task)] / (sum_FP + self.EachTaskTP[str(true_task)])
            else:
                task_prec = 0
                
            task_prec_list.append(task_prec)
            
        return task_prec_list


def test_AEClf_for_all_training_task( final_classes = 50, final_task = 1):

    cfg = Config()
    base_classes  = cfg.base_classes    
    increm_classes = cfg.incre_classes
    feature_dim = cfg.feature_dim
    branch = cfg.data_branch + '/feature_set'
    
    task_acc_list_dict = {}
    for cur_task in range(1, final_task+1):
        
        task_acc_list_dict.update({str(cur_task) : []})
    
        print("loading model...")
        cur_classes = base_classes + increm_classes * (cur_task-1)
        
        # cnn = Load_Model(cur_classes, cur_task, previous_epoch = previous_epoch)
        AE = Load_Model(cur_classes, cur_task, previous_epoch = previous_epoch, model_type = 'ae')
        
        
        model = AE.to(device)
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
                    
                    feature = feature.view(-1,feature_dim)
                    feature = feature.to(device)
                    label = label.to(device)                    
                             
                    output = model.encode(feature, inp_mode = "f")
                    # pred = model.classify(output)
                    pred = output
                 
                    final_pred = torch.argmax(pred, dim = 1)
                    final_pred = final_pred.view(label.shape)
                    acc_sample += torch.sum(final_pred == label).item()
                    total_sample += label.shape[0]
                    
            print("\ntest accuracy:",acc_sample/total_sample)
                
            task_acc_list_dict[str(cur_task)].append(acc_sample/total_sample)   
        
    return task_acc_list_dict


def test_BasicCNN_Prec( final_classes = 50, final_task = 1):
    
    task_prec_list_dict = {}
    for cur_task in range(1, final_task+1):
        
        task_prec_list_dict.update({str(cur_task) : []})
    
        print("loading model...")
        cur_classes = base_classes + increm_classes * (cur_task-1)
        cnn = Load_Model(cur_classes, cur_task, previous_epoch = previous_epoch)

        model = cnn.to(device)
        model.eval()


        root = './data'
        
        # test classes in task has seen  
        PrecCounter = TaskPrecision(cur_task)
        task_class_digit = []
        
        for test_task in range(1, cur_task+1):            
            task_class_digit.append(base_classes + (test_task - 1) * increm_classes)        
        
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
                             
                    output = model(feature, "f")
                 
                    final_pred = torch.argmax(output, dim = 1)
                    final_pred = final_pred.view(label.shape)
                    acc_sample += torch.sum(final_pred == label).item()
                    total_sample += label.shape[0]
                    
                    PrecCounter.ConutTP(test_task, acc_sample)
                    
                    #allocate FP in each task except self
                    for pred_task in range(1, cur_task+1):
                        if pred_task != test_task:

                            upper_class_digit = task_class_digit[pred_task-1]
                            lower_class_digit = task_class_digit[pred_task-1-1]
                            
                            task_wrong_sample = torch.sum(((final_pred < upper_class_digit) == (final_pred >= lower_class_digit))).item()
                            
                            PrecCounter.CountFP(test_task, pred_task, task_wrong_sample)
            
            # task_acc_list_dict[str(cur_task)].append(acc_sample/total_sample)
            print("\ntest accuracy:",acc_sample/total_sample)
            
        task_prec_list = PrecCounter.ComputeTaskPrecision()
        task_prec_list_dict[str(cur_task)] =  task_prec_list
        for true_task in range(1, cur_task+1):
            print("task {}:".format(true_task))
            print("test prec:{}".format(task_prec_list[true_task-1]))
        
        
    return task_prec_list_dict


def test_basic_cnn_for_all_training_task( final_classes = 50, final_task = 1):

    cfg = Config()
    base_classes  = cfg.base_classes    
    increm_classes = cfg.incre_classes
    feature_dim = cfg.feature_dim
    branch = cfg.data_branch + '/feature_set'
    
    task_acc_list_dict = {}
    for cur_task in range(1, final_task+1):
        
        task_acc_list_dict.update({str(cur_task) : []})
        print("training task", cur_task)
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
            
            print(" test task:{} / classes:{}".format(test_task, cur_classes))
            with torch.no_grad():

                acc_sample = 0
                total_sample = 0
                        
                for i, (feature, label) in enumerate(test_loader):
                    print('.', end = '')
                    
                    feature = feature.view(-1,feature_dim)
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

def test_sscnn_forSSAE(final_classes = 50, final_task = 1):
    
    insertion_check_list = [1,1,1]
    # kernel_size = 
    
    task_acc_list_dict = {}
    for train_task in range(1, final_task+1):
        
        task_acc_list_dict.update({str(train_task) : []})
    
        #load base model
        print("\nTraining task: {}".format(train_task)) 
        model_classes = base_classes + increm_classes * (train_task - 1)
        base_model = Load_Model(model_classes, train_task, previous_epoch = 0, model_type = 'ae')
        
        #load SS of task (each of 2~n)
        insert_check_list = [1,1,1,1,1] 
        ss_size_list = [2048, 1024, 1024, 512, 512]        
        en_ss_list = Load_EN_Kernel(insert_check_list, ss_size_list, train_task, training=False, previous_epoch = 0)

            
        #Combine SS and model
        model = SSAEwithFE(base_model, insert_check_list, ss_size_list)
        model.ss_set2 = en_ss_list
        model.eval()
        model = model.to(device)     
  
        for test_task in range(1, train_task+1):
            
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
                             
                    output = model.single_encode(feature, "f")
                 
                    final_pred = torch.argmax(output, dim = 1)
                    final_pred = final_pred.view(label.shape)
                    acc_sample += torch.sum(final_pred == label).item()
                    total_sample += label.shape[0]
                    
            print("\ntest accuracy:",acc_sample/total_sample)
                
            task_acc_list_dict[str(train_task)].append(acc_sample/total_sample)   
        
    return task_acc_list_dict
    
    
def test_sscnn_for_all_training_task(final_classes = 50, final_task = 1):
    
    insertion_check_list = [1,1,1]
    # kernel_size = 
    
    
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
