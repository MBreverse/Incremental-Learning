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


# # Import datetime class from datetime module
# from datetime import datetime
 
# # returns current date and time
# now = datetime.now()
# print(now)

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
feature_dim = 2048

#kernel hyper-parameter
insertion_check_list = [1,1,1]
kernel_size = [2048, 1024, 1024]

#dataset path
root = cfg.data_root
dataset_name = cfg.data_name
branch = cfg.data_branch


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

#hierachy test for all classes
def test_all_task(base_classes = 50, increm_classes = 5, final_task =2):
    task_acc_list_dict = {}
    #load end-training model
    for cur_task in range(1, final_task+1):
    
        task_acc_list_dict.update({str(cur_task) : []})
        
        print("\nTraining task: {}".format(cur_task))
        cur_classes = base_classes + increm_classes * (cur_task-1)

        print("loading model...")
        cnn = Load_Model(cur_classes, cur_task, previous_epoch = previous_epoch)

        print("loading kernel...")
        all_kernel_list = []
        # load all kernel
        for i in range(2, cur_task + 1):
            
            kernel_list1 = Load_Kernel( insertion_check_list, kernel_size, i, training=False, previous_epoch = previous_epoch)
                
            all_kernel_list.append(kernel_list1)

        #conbine model and kernel
        model = SS_Model( cnn, insertion_check_list, kernel_size, all_kernel_list )

        model = model.to(device)

        model.eval()

        # test classes in task has seen  
        for test_task in range(1, cur_task+1):

            dlabel = str(test_task)
            
            
            dataset_path = root + dataset_name + branch + '//split'+dlabel+'_dataset.pth.tar'    # 新任務的資料路徑
            
            dataset = torch.load(dataset_path)        
            
            test_loader = DataLoader(dataset['val_true'], batch_size = batch_size, shuffle = False)                   
            
            test_classes_low_bound = 0 + base_classes * (test_task>1) + increm_classes * (test_task-1)* (test_task>2)
            test_classes_upper_bound = base_classes + increm_classes * (test_task-1)
            print("test task:{} / classes:{}~{}".format(test_task, test_classes_low_bound, test_classes_upper_bound ))
            
            with torch.no_grad():

                acc_sample = 0
                total_sample = 0
                total_loss = 0
                
                t_count =0

                for i, (feature, label) in enumerate(test_loader):
                    print('.', end = '')
                    
                    feature = feature.view(-1,feature_dim)
                    feature = feature.to(device)
                    label = label.to(device)                    
                                        
                    out_list = model(feature, 'f')
                
                    classes = cur_classes

                    '''prediction for MKIL model'''
                    #initial  variable wrt predition result 
                    final_pred = torch.zeros(label[None,...].shape).to(device)
                    mask_id_list =[]
                    counter = 0
                    
                    # initial low bound class of each task on prediction ,as list like [-1, ..., (final classes-1) - increm_classes]
                    class_low_bound =[-1]+ [x * increm_classes + (base_classes-1) for x in range(final_task-1)]

                    for j in range(len(out_list)-1 ,-1, -1):
                        
                        output = out_list[j]
                        output = output[:, 0:classes]

                        class_pred = torch.argmax(output, dim = 1)

                        # check if the final class of predition mask
                        if mask_id_list!=[]:
                            for mid in mask_id_list:
                                class_pred[mid] = torch.zeros(mid.shape).to(torch.int64).to(device)

                        # mask that predict classes under task kernel class
                        mask = class_pred > class_low_bound[j] #(classes-5)
                        mask_id = torch.argwhere(mask==1)[:,0]

                        # check if the last class mask exists              
                        if mask_id.shape[0] != 0 :
                            mask_id_list.append(mask_id)
                            counter += mask_id.shape[0]
                        
                            
                        class_pred = class_pred * mask

                        classes = classes - increm_classes
                        
                        final_pred += class_pred[None,...]

                        if counter == batch_size:
                            break

                    final_pred = final_pred.view(label.shape)
                    acc_sample += torch.sum(final_pred == label).item()
                    total_sample += label.shape[0]
            
            task_acc_list_dict[str(cur_task)].append(acc_sample/total_sample)
            print("\ntest accuracy:",acc_sample/total_sample)
    return task_acc_list_dict
                                                                   
#hierachy test for all classes
def test_all_task_prec(base_classes = 50, increm_classes = 5, final_task =2, mode = "precision"):
    # task_acc_list_dict = {}
    task_prec_list_dict = {}
    #load end-training model
    for cur_task in range(1, final_task+1):
    
        task_prec_list_dict.update({str(cur_task) : []})
        
        print("\nTraining task: {}".format(cur_task))
        cur_classes = base_classes + increm_classes * (cur_task-1)

        print("loading model...")
        cnn = Load_Model(cur_classes, cur_task, previous_epoch = previous_epoch)

        print("loading kernel...")
        all_kernel_list = []
        # load all kernel
        for i in range(2, cur_task + 1):
            
            kernel_list1 = Load_Kernel( insertion_check_list, kernel_size, i, training=False, previous_epoch = previous_epoch)
                
            all_kernel_list.append(kernel_list1)

        #conbine model and kernel
        model = SS_Model( cnn, insertion_check_list, kernel_size, all_kernel_list )

        model = model.to(device)

        model.eval()

        # test classes in task has seen  
        PrecCounter = TaskPrecision(cur_task)
        task_class_digit = []
        for test_task in range(1, cur_task+1):
            
            task_class_digit.append(base_classes + (test_task - 1) * increm_classes)

        for test_task in range(1, cur_task+1):
            
            dlabel = str(test_task)
            
            
            dataset_path = root + dataset_name + branch + '//split'+dlabel+'_dataset.pth.tar'    # 新任務的資料路徑
            
            dataset = torch.load(dataset_path)        
            
            test_loader = DataLoader(dataset['val_true'], batch_size = batch_size, shuffle = False)                   
            
            test_classes_low_bound = 0 + base_classes * (test_task>1) + increm_classes * (test_task-1)* (test_task>2)
            test_classes_upper_bound = base_classes + increm_classes * (test_task-1)
            print("test task:{} / classes:{}~{}".format(test_task, test_classes_low_bound, test_classes_upper_bound ))                        
             
            
            with torch.no_grad():

                acc_sample = 0
                wro_sample = 0
                total_sample = 0
                total_loss = 0
                
                t_count =0

                for i, (feature, label) in enumerate(test_loader):
                    print('.', end = '')
                    
                    feature = feature.view(-1,feature_dim)
                    feature = feature.to(device)
                    label = label.to(device)                    
                                        
                    out_list = model(feature, 'f')
                
                    classes = cur_classes

                    '''prediction for MKIL model'''
                    #initial  variable wrt predition result 
                    final_pred = torch.zeros(label[None,...].shape).to(device)
                    mask_id_list =[]
                    counter = 0
                    
                    # initial low bound class of each task on prediction ,as list like [-1, ..., (final classes-1) - increm_classes]
                    class_low_bound =[-1]+ [x * increm_classes + (base_classes-1) for x in range(final_task-1)]

                    for j in range(len(out_list)-1 ,-1, -1):
                        
                        output = out_list[j]
                        output = output[:, 0:classes]

                        class_pred = torch.argmax(output, dim = 1)

                        # check if the final class of predition mask
                        if mask_id_list!=[]:
                            for mid in mask_id_list:
                                class_pred[mid] = torch.zeros(mid.shape).to(torch.int64).to(device)

                        # mask that predict classes under task kernel class
                        mask = class_pred > class_low_bound[j] #(classes-5)
                        mask_id = torch.argwhere(mask==1)[:,0]

                        # check if the last class mask exists              
                        if mask_id.shape[0] != 0 :
                            mask_id_list.append(mask_id)
                            counter += mask_id.shape[0]
                        
                            
                        class_pred = class_pred * mask

                        classes = classes - increm_classes
                        
                        final_pred += class_pred[None,...]

                        if counter == batch_size:
                            break

                    final_pred = final_pred.view(label.shape)
                    acc_sample += torch.sum(final_pred == label).item()
                    total_sample += label.shape[0]

                    PrecCounter.ConutTP(test_task, acc_sample)
                    
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
    
def test_all_task_with_single_task_model(base_classes = 50, increm_classes = 5, final_task =2):
    
    #load end-training model    
    print("\nTraining task: {}".format(final_task))
    cur_classes = base_classes + increm_classes * (final_task-1)

    print("loading model...")
    cnn = Load_Model(cur_classes, final_task, previous_epoch = previous_epoch)

    print("loading kernel...")
    all_kernel_list = []
    # load all kernel
    for i in range(2, final_task + 1):
        
        kernel_list1 = Load_Kernel( insertion_check_list, kernel_size, i, training=False, previous_epoch = previous_epoch)
            
        all_kernel_list.append(kernel_list1)

    #conbine model and kernel
    model = SS_Model( cnn, insertion_check_list, kernel_size, all_kernel_list )

    model = model.to(device)

    model.eval()

    # test classes in task has seen  
    for test_task in range(1, final_task+1):

        dlabel = str(test_task)
        
        # cur_classes = base_classes + increm_classes * (test_task-1)
        
        dataset_path = root + dataset_name + branch + '//split'+dlabel+'_dataset.pth.tar'    # 新任務的資料路徑
        
        dataset = torch.load(dataset_path)        
        
        test_loader = DataLoader(dataset['val_true'], batch_size = batch_size, shuffle = False)                   
        
        test_classes_low_bound = 0 + base_classes * (test_task>1) + increm_classes * (test_task-1)* (test_task>2)
        test_classes_upper_bound = base_classes + increm_classes * (test_task-1)
        print("test task:{} / classes:{}~{}".format(test_task, test_classes_low_bound, test_classes_upper_bound ))
        
        with torch.no_grad():

            acc_sample = 0
            total_sample = 0
            total_loss = 0
            
            t_count =0

            for i, (feature, label) in enumerate(test_loader):
                print('.', end = '')
                
                feature = feature.view(-1,feature_dim)
                feature = feature.to(device)
                label = label.to(device)                    
                         
                out_list = model(feature, 'f')
            
                classes = cur_classes

                #initial  variable wrt predition result 
                final_pred = torch.zeros(label[None,...].shape).to(device)
                mask_id_list =[]
                counter = 0
                
                # initial low bound class of each task on prediction ,as list like [-1, ..., (final classes-1) - increm_classes]
                class_low_bound =[-1]+ [x * increm_classes + (base_classes-1) for x in range(final_task-1)]

                for j in range(len(out_list)-1 ,-1, -1):
                    
                    output = out_list[j]
                    output = output[:, 0:classes]

                    class_pred = torch.argmax(output, dim = 1)

                    # check if the final class of predition mask
                    if mask_id_list!=[]:
                        for mid in mask_id_list:
                            class_pred[mid] = torch.zeros(mid.shape).to(torch.int64).to(device)

                    # mask that predict classes under task kernel class
                    mask = class_pred > class_low_bound[j] #(classes-5)
                    mask_id = torch.argwhere(mask==1)[:,0]

                    # check if the last class mask exists              
                    if mask_id.shape[0] != 0 :
                        mask_id_list.append(mask_id)
                        counter += mask_id.shape[0]
                    
                        
                    class_pred = class_pred * mask

                    classes = classes - increm_classes
                    
                    final_pred += class_pred[None,...]

                    if counter == batch_size:
                        break

                final_pred = final_pred.view(label.shape)
                acc_sample += torch.sum(final_pred == label).item()
                total_sample += label.shape[0]
    
        print("\ntest accuracy:",acc_sample/total_sample)
                
def test_each_task_with_single_task_model(base_classes = 50, increm_classes = 5, final_task =2):
    
    #load end-training model    
    print("\nTraining task: {}".format(final_task))
    cur_classes = base_classes + increm_classes * (final_task-1)

    print("loading model...")
    cnn = Load_Model(cur_classes, final_task, previous_epoch = previous_epoch)

    print("loading kernel...")
    all_kernel_list = []
    # load all kernel
    for i in range(2, final_task + 1):
        
        kernel_list1 = Load_Kernel( insertion_check_list, kernel_size, i, training=False, previous_epoch = previous_epoch)
            
        all_kernel_list.append(kernel_list1)

    #conbine model and kernel
    model = SS_Model( cnn, insertion_check_list, kernel_size, all_kernel_list )

    model = model.to(device)

    model.eval()

    # test classes single task  
    test_task = final_task

    dlabel = str(test_task)
    
    # cur_classes = base_classes + increm_classes * (test_task-1)
    
    dataset_path = root + dataset_name + branch + '//split'+dlabel+'_dataset.pth.tar'    # 新任務的資料路徑
    
    dataset = torch.load(dataset_path)        
    
    test_loader = DataLoader(dataset['val_true'], batch_size = batch_size, shuffle = False)                   
    
    test_classes_low_bound = 0 + base_classes * (test_task>1) + increm_classes * (test_task-1)* (test_task>2)
    test_classes_upper_bound = base_classes + increm_classes * (test_task-1)
    print("test task:{} / classes:{}~{}".format(test_task, test_classes_low_bound, test_classes_upper_bound ))
    
    with torch.no_grad():

        acc_sample = 0
        total_sample = 0
        total_loss = 0
        
        t_count =0

        for i, (feature, label) in enumerate(test_loader):
            print('.', end = '')
            
            feature = feature.view(-1,feature_dim)
            feature = feature.to(device)
            label = label.to(device)                    
                     
            out_list = model(feature, 'f')
        
            classes = cur_classes

            #initial  variable wrt predition result 
            final_pred = torch.zeros(label[None,...].shape).to(device)
            mask_id_list =[]
            counter = 0
            
            # initial low bound class of each task on prediction ,as list like [-1, ..., (final classes-1) - increm_classes]
            class_low_bound =[-1]+ [x * increm_classes + (base_classes-1) for x in range(final_task-1)]

            for j in range(len(out_list)-1 ,-1, -1):
                
                output = out_list[j]
                output = output[:, 0:classes]

                class_pred = torch.argmax(output, dim = 1)

                # check if the final class of predition mask
                if mask_id_list!=[]:
                    for mid in mask_id_list:
                        class_pred[mid] = torch.zeros(mid.shape).to(torch.int64).to(device)

                # mask that predict classes under task kernel class
                mask = class_pred > class_low_bound[j] #(classes-5)
                mask_id = torch.argwhere(mask==1)[:,0]

                # check if the last class mask exists              
                if mask_id.shape[0] != 0 :
                    mask_id_list.append(mask_id)
                    counter += mask_id.shape[0]
                
                    
                class_pred = class_pred * mask

                classes = classes - increm_classes
                
                final_pred += class_pred[None,...]

                if counter == batch_size:
                    break

            final_pred = final_pred.view(label.shape)
            acc_sample += torch.sum(final_pred == label).item()
            total_sample += label.shape[0]

    print("\ntest accuracy:",acc_sample/total_sample)
            
def test_all_task_with_final_task_model(base_classes = 50, increm_classes = 5, final_task = 3):
    #load end-training model    
    print("\nTraining task: {}".format(final_task))
    cur_classes = base_classes + increm_classes * (final_task-1)

    print("loading model...")
    cnn = Load_Model(cur_classes, final_task, previous_epoch = previous_epoch)

    print("loading kernel...")
    all_kernel_list = []
    # load all kernel
    for i in range(2, final_task + 1):
        
        kernel_list1 = Load_Kernel( insertion_check_list, kernel_size, i, training=False, previous_epoch = previous_epoch)
            
        all_kernel_list.append(kernel_list1)

    #conbine model and kernel
    model = SS_Model( cnn, insertion_check_list, kernel_size, all_kernel_list )

    model = model.to(device)

    model.eval()

    # test classes in task has seen  
    for test_task in range(1, final_task+1):

        dlabel = str(test_task)
        
        # cur_classes = base_classes + increm_classes * (test_task-1)
        
        dataset_path = root + dataset_name + branch + '//split'+dlabel+'_dataset.pth.tar'    # 新任務的資料路徑
        
        dataset = torch.load(dataset_path)        
        
        test_loader = DataLoader(dataset['val_true'], batch_size = batch_size, shuffle = False)                   
        
        test_classes_low_bound = 0 + base_classes * (test_task>1) + increm_classes * (test_task-1)* (test_task>2)
        test_classes_upper_bound = base_classes + increm_classes * (test_task-1)
        print("test task:{} / classes:{}~{}".format(test_task, test_classes_low_bound, test_classes_upper_bound ))
        
        with torch.no_grad():

            acc_sample = 0
            total_sample = 0
            total_loss = 0
            
            t_count =0

            for i, (feature, label) in enumerate(test_loader):
                print('.', end = '')
                
                feature = feature.view(-1,feature_dim)
                feature = feature.to(device)
                label = label.to(device)                    
                         
                out_list = model(feature, 'f')
            

                _,final_pred = torch.max(out_list[-1], dim = 1)
                final_pred = final_pred.view(label.shape)
                acc_sample += torch.sum(final_pred == label).item()
                total_sample += label.shape[0]
    
        print("\ntest accuracy:",acc_sample/total_sample)