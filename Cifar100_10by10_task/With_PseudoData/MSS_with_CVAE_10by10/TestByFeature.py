import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import vgg16, resnet50
from torchvision.utils import save_image

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

from LossFunction import *
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

class HierachyPrediction():
    def __init__(self, cur_classes):
        
        self.cur_classes = cur_classes
        
        cfg = Config()
        self.incre_classes = cfg.incre_classes
        self.base_classes = cfg.base_classes
            
    def predict(self, pred_list):

        #initial variable wrt predition result 
        final_pred = torch.zeros(pred_list[0].shape[0]).to(device)
        final_task = len(pred_list)
        cur_classes = self.cur_classes
        
        mask_id_list =[]
        counter = 0
        
        # initial low bound class of each task on prediction ,as list like [-1, ..., (final classes-1) - increm_classes]
        class_low_bound =[-1]+ [x * self.incre_classes  + (self.base_classes - 1) for x in range(final_task-1)]

        for j in range(len(pred_list)-1 ,-1, -1):
            
            output = pred_list[j]
            output = output[:, 0:cur_classes]

            class_pred = torch.argmax(output, dim = 1)

            # check if the final class of predition mask
            if mask_id_list!=[]:
                for mid in mask_id_list:
                    class_pred[mid] = torch.zeros(mid.shape).to(torch.int64).to(device)

            # mask that predict classes under task kernel class
            mask = class_pred > class_low_bound[j] #(classes-5)
            mask_id = torch.argwhere(mask==1)[:,0]

            # check if the previous class mask exists              
            if mask_id.shape[0] != 0 :
                mask_id_list.append(mask_id)
                counter += mask_id.shape[0]
                            
            class_pred = class_pred * mask

            cur_classes = cur_classes - self.incre_classes
            
            final_pred += class_pred#[None,...]

            if counter == batch_size:
                break
        
        return final_pred

#hierachy test for all classes
def test_all_task(final_task =2):
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
def test_all_task_prec(final_task =2, mode = "precision"):
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
        FPsbyTask = {}
        TPbyTask = {}
        for test_task in range(1, cur_task+1):
            
            FPsbyTask.update({str(test_task):[]})
            TPbyTask.update({str(test_task):[]})

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
                    # wro_sample += torch.sum(final_pred != label).item()
                    total_sample += label.shape[0]
                    
                    # for 
                    # no_regular_pred = (final_pred >= cur_classes) == (final_pred < (cur_classes-increm_classes))
                    # FPsbyTask[str(test_task)].append( torch.sum(no_regular_pred).item())
            
            task_acc_list_dict[str(cur_task)].append(acc_sample/total_sample)
            print("\ntest accuracy:",acc_sample/total_sample)
            
    return task_acc_list_dict                       

def test_all_task_with_single_task_model( final_task =2):
    
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
                
def test_each_task_with_single_task_model( final_task =2):
    
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
            
def test_all_task_with_final_task_model( final_task = 3):

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


def test_all_task_with_all_SSModel(final_task = 2, mode = "prec&acc"):

    cfg = Config()
    base_classes  = cfg.base_classes    
    increm_classes = cfg.incre_classes
    insertion_check_list = cfg.insert_check_list
    kernel_size = cfg.ss_sizes
    feature_dim = cfg.feature_dim
    branch = cfg.data_branch #+ "/feature_set"
    
    task_acc_list_dict = {}
    task_prec_list_dict = {}
    #load end-training model
    for cur_task in range(1, final_task+1):
        
        task_acc_list_dict.update({str(cur_task) : []})    
        task_prec_list_dict.update({str(cur_task) : []})
        
        print("\nTraining task: {}".format(cur_task))
        cur_classes = base_classes + increm_classes * (cur_task-1)

        print("loading model...")
        cnn = Load_Model(cur_classes, cur_task, previous_epoch = previous_epoch)

        print("loading kernels...")
        all_kernel_list = Load_All_Task_SS(insertion_check_list, kernel_size, cur_task, training = False)

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

                for i, (feature, label) in enumerate(test_loader):
                    print('.', end = '')
                    
                    feature = feature.view(-1,feature_dim)
                    feature = feature.to(device)
                    label = label.to(device)                    
                                        
                    out_list = model(feature, 'f')

                    HPredictor = HierachyPrediction(cur_classes)                
                    final_pred = HPredictor.predict(out_list)                    

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
            
            task_acc_list_dict[str(cur_task)].append(acc_sample/total_sample)
            print("\ntest accuracy:",acc_sample/total_sample)
            
        task_prec_list = PrecCounter.ComputeTaskPrecision()
        task_prec_list_dict[str(cur_task)] =  task_prec_list
        for true_task in range(1, cur_task+1):
            print("task {}:".format(true_task))
            print("test prec:{}".format(task_prec_list[true_task-1]))

    if mode == "prec&acc":
        return task_acc_list_dict, task_prec_list_dict
    if mode == "prec":
        return task_prec_list_dict    
    if mode == "acc":
        return task_acc_list_dict  
    
def test_all_task_with_single_SSmodel(cur_task = 2, mode = "prec&acc"):

    cfg = Config()
    base_classes  = cfg.base_classes    
    increm_classes = cfg.incre_classes
    insertion_check_list = cfg.insert_check_list
    kernel_size = cfg.ss_sizes
    feature_dim = cfg.feature_dim
    branch = cfg.data_branch + "/feature_set"
    
    task_acc_list_dict = {}
    task_prec_list_dict = {}
    #load end-training model

        
    task_acc_list_dict.update({str(cur_task) : []})    
    task_prec_list_dict.update({str(cur_task) : []})
    
    print("\nTraining task: {}".format(cur_task))
    cur_classes = base_classes + increm_classes * (cur_task-1)

    print("loading model...")
    cnn = Load_Model(cur_classes, cur_task, previous_epoch = previous_epoch)

    print("loading kernels...")
    all_kernel_list = Load_All_Task_SS(insertion_check_list, kernel_size, cur_task, training = False)

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

            for i, (feature, label) in enumerate(test_loader):
                print('.', end = '')
                
                feature = feature.view(-1,feature_dim)
                feature = feature.to(device)
                label = label.to(device)                    
                                    
                out_list = model(feature, 'f')

                HPredictor = HierachyPrediction(cur_classes)                
                final_pred = HPredictor.predict(out_list)                    

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
        
        task_acc_list_dict[str(cur_task)].append(acc_sample/total_sample)
        print("\ntest accuracy:",acc_sample/total_sample)
        
    task_prec_list = PrecCounter.ComputeTaskPrecision()
    task_prec_list_dict[str(cur_task)] =  task_prec_list
    for true_task in range(1, cur_task+1):
        print("task {}:".format(true_task))
        print("test prec:{}".format(task_prec_list[true_task-1]))

    if mode == "prec&acc":
        return task_acc_list_dict, task_prec_list_dict
    if mode == "prec":
        return task_prec_list_dict    
    if mode == "acc":
        return task_acc_list_dict  
       
def test_all_task_with_final_SSmodel(final_task = 3):
    
    cfg = Config()
    base_classes  = cfg.base_classes    
    increm_classes = cfg.incre_classes
    insertion_check_list = cfg.insert_check_list
    kernel_size = cfg.ss_sizes
    feature_dim = cfg.feature_dim
    branch = cfg.data_branch + "/feature_set"
    
    #load end-training model    
    print("\nTraining task: {}".format(final_task))
    cur_classes = base_classes + increm_classes * (final_task-1)

    print("loading model...")
    cnn = Load_Model(cur_classes, final_task, previous_epoch = previous_epoch)

    print("loading kernels...")
    all_kernel_list = Load_All_Task_SS(insertion_check_list, kernel_size, cur_task, training = False)

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

def test_recover_result(task):

    record_root = cfg.record_root + cfg.recover_folder
    
    if os.path.exists(record_root) ==False:
        os.mkdir(record_root)
           
    model = Load_AE()       
    model.to(device)
    model.eval()
    
    for t in range(1, task+1):

        dlabel = str(t)        
        
        dataset_path = root + dataset_name + branch + '//split'+dlabel+'_dataset.pth.tar'    # 新任務的資料路徑
        
        dataset = torch.load(dataset_path)        
        
        test_loader = DataLoader(dataset['val_true'], batch_size = 1, shuffle = True) # random sample

        for i, (img, label) in enumerate(test_loader):                         
            
            data = torch.flatten(img)
            data = data.view(-1,cfg.input_dim)
            data = data.to(device)
            label = label.to(device)

            # forward           
            _, _, _, x_head = model(data)
            
            re_img = x_head.view(cfg.data_size)
            re_img = re_img.to("cpu")

            img = img.view(cfg.data_size)
            
            comb_img = torch.cat([re_img, img], dim = 2)
                        
            torchvision.utils.save_image(comb_img, fp = record_root + '/t{}_{}th.png'.format(t, i))
            
            if i == 19:
                break                               
     
def test_rand_gen_result():

    # data_root = './data'
    # branch = ''   
    # dataset_name = '/Mnist_Split'
    # record_root = './record/RecovResult'
    record_root = cfg.record_root + cfg.generate_folder
    
    if os.path.exists(record_root) ==False:
        os.mkdir(record_root)
           
    model = Load_AE()       
    model.to(device)
    model.eval()
    
    # forward        
    
    x_head = model.generate(64)
    
    re_img = x_head.view(-1,1,28,28)
    # re_img = re_img * 255.
    re_img = re_img.to("cpu")
                
    torchvision.utils.save_image(re_img, fp = record_root + '/RandGenerate.png')

'''test cvae revovery error'''
def test_cvae_for_all_task(final_task):
    
    CIter = Incre_Class_Iter(cfg.base_classes, cfg.incre_classes, cfg.final_classes)
    
    task_error_dict = {}
    for train_task in range(1, final_task+1):
        print("train task ", train_task)
        
        s_digit, e_digit = CIter.get_cur_class_digits(train_task)
        cur_classes = e_digit + 1
        
        model = CVAE(cfg.input_dim, cfg.z_dim, cur_classes).to(device)  
        
        model_file = record_root + '/task{}_cvae_model.pth.tar'.format(train_task)
        model_dict = torch.load(model_file)
        model.load_state_dict(model_dict)
        model.eval()
        
        task_error_dict.update({str(train_task) : []})
        for test_task in range(1, train_task+1):
            
            test_loader = get_Cifar_task_dataloder(data_root = cfg.data_root, branch = cfg.data_branch, task = test_task, train = False, batch_size = 1024)
            
            total_size = 0
            error_accl_sum = 0
            with torch.no_grad():
                for i, (data, label) in enumerate(test_loader):
                
                    data = data.view(-1, cfg.input_dim).to(device)
                    label = label.to(device)
                    
                    onehot = one_hot(label, cur_classes)
                    x_head, mu, var = model(data, onehot)
                    
                    error = (x_head.view(-1, cfg.input_dim) - data)**2
                    
                    error_accl_sum += torch.sum(error).item()
                    total_size  += label.size(0)
                
                total_error_avg = error_accl_sum/total_size
                
                task_error_dict[str(train_task)].append(total_error_avg)
        
            print(" task{} error: {}".format(test_task, total_error_avg))
    
    return task_error_dict
 
'''test cvae revovery of img'''
def test_cvae_recover_img(final_task):

    CIter = Incre_Class_Iter(cfg.base_classes, cfg.incre_classes, cfg.final_classes)
    
    for train_task in range(1, final_task+1):
        print("train task ",train_task)
    
        model = CVAE(28*28, cfg.z_dim, 10).to(device)  
        
        model_file = record_root + '/task{}_cvae_model.pth.tar'.format(train_task)
        model_dict = torch.load(model_file)
        model.load_state_dict(model_dict)
        model.eval()
        
        s_digit, e_digit = CIter.get_cur_class_digits(train_task)
        class_check_list  = [0] * (e_digit + 1)

        
        for test_task in range(1, train_task+1):
            print(" test task ",test_task)
        
            test_loader = get_Mnist_task_dataloder(data_root = cfg.data_root, branch = cfg.data_branch, task = test_task, train = False, batch_size = 10)
            

            with torch.no_grad():
                for i, (data, label) in enumerate(test_loader):
                
                    data = data.view(-1, cfg.input_dim).to(device)
                    label = label.to(device)
                    
                    # check if class is selected
                    for j, check in enumerate(class_check_list):
                        
                        start_class_digit = j
                        
                        if check == 0 and start_class_digit == label[0]:

                            class_check_list[j] = 1
                        
                            onehot = one_hot(label, cfg.final_classes)
                            xhead, mu, var = model(data, onehot)
                            
                            cat_x_xhead = torch.cat([data[0].view(cfg.data_size), xhead[0].view(cfg.data_size)], dim = 1)         

                            if os.path.exists(cfg.recover_folder) == False:
                                os.mkdir(cfg.recover_folder)   
                                
                            if os.path.exists(cfg.recover_folder + "/task{}".format(train_task)) == False:
                                os.mkdir(cfg.recover_folder + "/task{}".format(train_task))                                 
                                
                            save_image(cat_x_xhead, cfg.recover_folder + "/task{}".format(train_task) + '/task{}_c{}_recover'.format(test_task, start_class_digit) + '.png')                            

'''test cvae generation of img'''
def test_cvae_generate_img(final_task):

    CIter = Incre_Class_Iter(cfg.base_classes, cfg.incre_classes, cfg.final_classes)
    for train_task in range(1, final_task + 1):
    
        model = CVAE(28*28, cfg.z_dim, 10).to(device)          
        model_file = record_root + '/task{}_cvae_model.pth.tar'.format(train_task)
        model_dict = torch.load(model_file)
        model.load_state_dict(model_dict) 
        print("-"*10)
        for target_task in range(1, train_task + 1):
        
            s_digit, e_digit = CIter.get_cur_class_digits(target_task)
            
            end_class_digit = e_digit              

            with torch.no_grad():       
                
                for i in range(s_digit , e_digit + 1):
                    print(i)
                    label = torch.as_tensor([i])
                    
                    onehot = one_hot(label, cfg.final_classes).to(device)
                    
                    sample = torch.randn(1, cfg.z_dim).to(device)
                                        
                    sample = model.decode(sample, onehot)     

                    if os.path.exists(cfg.generate_folder) == False:
                        os.mkdir(cfg.generate_folder)   
                        
                    if os.path.exists(cfg.generate_folder + "/task{}".format(train_task)) == False:
                        os.mkdir(cfg.generate_folder + "/task{}".format(train_task))                           
                
                    save_image(sample.view(1, 28, 28), cfg.generate_folder + "/task{}".format(train_task) + '/task{}_c{}sample'.format(target_task, i) + '.png')
    
if __name__ == "__main__":

    print("successfully")