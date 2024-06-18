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

#specified testing hyper-parameter
previous_epoch = 0   
batch_size = 128

#dataset path
cfg = Config()

root = cfg.data_root
dataset_name = cfg.dataset_name
branch = cfg.data_branch 

incre_classes = cfg.incre_classes
base_classes = cfg.base_classes

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
def test_all_task_recall_on_HierachyPred(final_task = 2):
    
    # hyper - parameter
    base_classes  = cfg.base_classes    
    increm_classes = cfg.incre_classes
    insertion_check_list = cfg.insert_check_list
    kernel_size = cfg.ss_sizes    
    feature_dim = cfg.feature_dim
    # branch = cfg.data_branch + '/feature_set'
    
    task_acc_list_dict = {}    
    #load end-training model
    for cur_task in range(1, final_task+1):
    
        task_acc_list_dict.update({str(cur_task) : []})
        
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

        # test classes in task has seen in training  
        for test_task in range(1, cur_task+1):

            dlabel = str(test_task)
            
            dataset_path = root + dataset_name + branch + '//split'+dlabel+'_dataset.pth.tar'               
            dataset = torch.load(dataset_path)                    
            test_loader = DataLoader(dataset['val_true'], batch_size = batch_size, shuffle = False)                   
            
            # testing classes digits bound 
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

                    HPredictor = HierachyPrediction(cur_classes)                
                    final_pred = HPredictor.predict(out_list)
                    
                    final_pred = final_pred.view(label.shape)
                    acc_sample += torch.sum(final_pred == label).item()
                    total_sample += label.shape[0]
            
            task_acc_list_dict[str(cur_task)].append(acc_sample/total_sample)
            print("\ntest accuracy:",acc_sample/total_sample)
            
    return task_acc_list_dict
                                                                   
#hierachy test for all classes
def test_all_task_with_all_SSModel(final_task = 2, mode = "prec&acc"):

    cfg = Config()
    base_classes  = cfg.base_classes    
    increm_classes = cfg.incre_classes
    insertion_check_list = cfg.insert_check_list
    kernel_size = cfg.ss_sizes
    feature_dim = cfg.feature_dim
    branch = cfg.data_branch
    
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
    branch = cfg.data_branch 
    
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
    branch = cfg.data_branch
    
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

#hierachy test for all classes
def test_all_task_BySSAE(final_task =2):

    cfg = Config()
    base_classes  = cfg.base_classes    
    increm_classes = cfg.incre_classes
    insert_check_list = cfg.insert_check_dict["en"]
    ss_size_list = cfg.ss_size_dict["en"]
    feature_dim = cfg.feature_dim
    branch = cfg.data_branch    
    
    task_acc_list_dict = {}
    
    #load train task model
    for train_task in range(1, final_task+1):   
        # task_acc_list_dict.update({str(cur_task) : []})    
        task_prec_list_dict.update({str(cur_task) : []})
                
        #load base model
        print("\nTraining task: {}".format(train_task)) 
        model_classes = base_classes + increm_classes * (train_task - 1)
        base_model = Load_Model(model_classes, train_task, previous_epoch = 0, model_type = 'ae')
        
        #load SS of all task (2~n)
        all_en_ss_list = []
        for t in range(2, train_task+1):
            en_ss_list = Load_EN_Kernel(insert_check_list, ss_size_list, t, training=False, previous_epoch = 0)
            all_en_ss_list.append(en_ss_list)
            
        #Combine SS and model
        model = SSAEwithFE(base_model, insert_check_list, ss_size_list, all_en_ss_list)
        model.eval()
        model.to(device)
        
        # load dataset for test task
        for test_task in range(1, train_task+1):
                
            dataset_path = root + '/Cifar100_split'+ branch +'//split'+ str(test_task) +'_dataset.pth.tar'    
            
            test_dataset = torch.load(dataset_path)  
            test_loader = DataLoader(test_dataset['val_true'], batch_size = batch_size, shuffle = False) 

            # indicate test task and class
            test_classes_low_bound = 0 + base_classes * (test_task>1) + increm_classes * (test_task-1)* (test_task>2)
            test_classes_upper_bound = base_classes + increm_classes * (test_task-1)
            print("test task:{} / classes:{}~{}".format(test_task, test_classes_low_bound, test_classes_upper_bound ))
            
            #pass data to model
            with torch.no_grad():

                acc_sample = 0
                total_sample = 0

                for i, (feature, label) in enumerate(test_loader):            
            
                    print('.', end = '')
                    
                    feature = feature.view(-1,feature_dim)
                    feature = feature.to(device)
                    label = label.to(device)                    
                                        
                    out_list = model(feature, 'f', out_mode = 'encode')
                
                    HPredictor = HierachyPrediction(model_classes)                
                    final_pred = HPredictor.predict(out_list)                     

                    final_pred = final_pred.view(label.shape)
                    acc_sample += torch.sum(final_pred == label).item()
                    total_sample += label.shape[0]
            
            task_acc_list_dict[str(train_task)].append(acc_sample/total_sample)
            print("\ntest accuracy:",acc_sample/total_sample)    

     
    return task_acc_list_dict

#hierachy test for all classes
def test_all_task_with_all_SSAE( final_task =2, mode = "prec&acc"):

    cfg = Config()
    base_classes  = cfg.base_classes    
    increm_classes = cfg.incre_classes
    insert_check_list = cfg.insert_check_dict["en"]
    ss_size_list = cfg.ss_size_dict["en"]
    feature_dim = cfg.feature_dim
    branch = cfg.data_branch    
    
    
    task_acc_list_dict = {}
    task_prec_list_dict = {}
    #load train task model
    for train_task in range(1, final_task+1):   
        
        task_acc_list_dict.update({str(cur_task) : []})    
        task_prec_list_dict.update({str(train_task) : []})
        
        #load base model
        print("\nTraining task: {}".format(train_task)) 
        model_classes = base_classes + increm_classes * (train_task - 1)
        base_model = Load_Model(model_classes, train_task, previous_epoch = 0, model_type = 'ae')
        
        #load SS of all task (2~n)
        all_en_ss_list = []
        for t in range(2, train_task+1):
            en_ss_list = Load_EN_Kernel(insert_check_list, ss_size_list, t, training=False, previous_epoch = 0)
            all_en_ss_list.append(en_ss_list)
            
        #Combine SS and model
        model = SSAEwithFE(base_model, insert_check_list, ss_size_list, all_en_ss_list)
        model.eval()
        model.to(device)

        # test classes in task has seen  
        PrecCounter = TaskPrecision(train_task)
        task_class_digit = []
        for test_task in range(1, train_task+1):
            
            task_class_digit.append(base_classes + (test_task - 1) * increm_classes)

        for test_task in range(1, train_task+1):
            
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

                for i, (feature, label) in enumerate(test_loader):
                    print('.', end = '')
                    
                    feature = feature.view(-1,feature_dim)
                    feature = feature.to(device)
                    label = label.to(device)                    
                                        
                    out_list = model(feature, 'f', out_mode = 'encode')
                
                    HPredictor = HierachyPrediction(model_classes)                
                    final_pred = HPredictor.predict(out_list)                    


                    final_pred = final_pred.view(label.shape)
                    acc_sample += torch.sum(final_pred == label).item()
                    total_sample += label.shape[0]

                    PrecCounter.ConutTP(test_task, acc_sample)
                    
                    for pred_task in range(1, train_task+1):
                        if pred_task != test_task:

                            upper_class_digit = task_class_digit[pred_task-1]
                            lower_class_digit = task_class_digit[pred_task-1-1]
                            
                            task_wrong_sample = torch.sum(((final_pred < upper_class_digit) == (final_pred >= lower_class_digit))).item()
                            
                            PrecCounter.CountFP(test_task, pred_task, task_wrong_sample)
            
            task_acc_list_dict[str(cur_task)].append(acc_sample/total_sample)
            print("\ntest accuracy:",acc_sample/total_sample)
            
        task_prec_list = PrecCounter.ComputeTaskPrecision()
        task_prec_list_dict[str(train_task)] =  task_prec_list
        for true_task in range(1, train_task+1):
            print("task {}:".format(true_task))
            print("test prec:{}".format(task_prec_list[true_task-1]))

    if mode == "prec&acc":
        return task_acc_list_dict, task_prec_list_dict
    if mode == "prec":
        return task_prec_list_dict    
    if mode == "acc":
        return task_acc_list_dict  
        
    return task_prec_list_dict
    
def test_all_task_recover_error_ByAE( final_task = 3):

    cfg = Config()
    base_classes  = cfg.base_classes    
    increm_classes = cfg.incre_classes
    insert_check_dict = cfg.insert_check_dict
    ss_size_dict = cfg.ss_size_dict
    feature_dim = cfg.feature_dim
    branch = cfg.data_branch  
    
    train_task_error_dict = {}
    for train_task in range(1, final_task+1):
        
        model_class = base_classes + increm_classes * (train_task-1)
        base_model = Load_Model(model_class, train_task, model_type = 'ae')

        task_error_statistic = {}
        for test_task in range(1, train_task+1):

            insertion_check_list = insert_check_dict["en"]
            kernel_size = ss_size_dict["en"]
            en_ss_list2 = Load_EN_Kernel(insertion_check_list, kernel_size, test_task, training=False)
            
            insertion_check_list = insert_check_dict["de"]
            de_ss_sizes = ss_size_dict["de"]
            de_ss_list2 = Load_DE_Kernel(insertion_check_list, de_ss_sizes, test_task, training = False)           
                        

            #conbine model and kernel
            if en_ss_list2 == None and de_ss_list2 == None:
                
                SSAE = base_model            
            else:
                SSAE = SSAEwithFE(base_model, insertion_check_list, kernel_size)
                SSAE.en_ss_set2 = en_ss_list2
                SSAE.de_ss_set2 = de_ss_list2 

            SSAE = SSAE.to(device)       
        

            task_error_statistic.update({str(test_task): {"mean" : 0 , "std" : 0} })
            
            dlabel = str(test_task)
            
            dataset_path = root + dataset_name + branch + '//split'+dlabel+'_dataset.pth.tar'    # 新任務的資料路徑
            
            dataset = torch.load(dataset_path)        
            
            test_loader = DataLoader(dataset['val_true'], batch_size = 1, shuffle = False) 

            #feed data
            E_list = []
            # count = 0
            for j, (feature, label) in enumerate(test_loader):
                
                feature = feature.to(device)
                label = label.to(device)
                
                feature = feature.view(-1, feature_dim)
                

                if test_task==1:
                    z_code = SSAE.encode(feature, 'f') 
                   
                else:
                    z_code = SSAE.single_encode(feature, 'f') 
                
                # cover the new weight influence
                if test_task < train_task:
                    zero_padding = torch.zeros(1, increm_classes*(train_task - test_task)).to(device)
                    z_code[:,-increm_classes*(train_task - test_task):] = zero_padding
                    # z_code = torch.cat([z_code, zero_padding], dim = 1)  
                
                recon_x = SSAE.decode(z_code)  
                
                error = torch.sum((recon_x - feature)**2)
                E_list.append(torch.unsqueeze(error,0))
                
            E_record = torch.cat(E_list, dim = 0)
            task_error_statistic[str(test_task)]["mean"] = torch.mean(E_record).item()
            task_error_statistic[str(test_task)]["std"] = torch.std(E_record).item()
        
        train_task_error_dict.update({str(train_task): task_error_statistic})
        
    return train_task_error_dict
                