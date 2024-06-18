#train by feature
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from torchvision.models import vgg16, resnet50

import matplotlib.pyplot as plt
from PIL import Image
import numpy as np

import os
import sys
import os.path
import json

from ModelCollection import *
from DatasetProcess import *
from LossFunction import *
from SupportFunction import *
from Track import LossTracker, AccTracker
from Record import SaveJson
from Config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
record_root = cfg.record_root 
# record_root = './record'

if os.path.exists(record_root) ==False:
    os.mkdir(record_root)

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

def base_training(model, train_loader, epochs): 
   
    model = model.to(device) 
    model.train()    
    model.FE.eval()  
    
    epochs = 15  
    lr = 0.001
         
    para_dict = [{'params': model.fc1.parameters()},{'params': model.fc2.parameters()}]

    opt = optim.Adam(para_dict, lr, weight_decay =  0.0005, amsgrad = True)
    
    loss_names = ["ce"] 
    ltker = LossTracker(num_loss = 1 , loss_names = loss_names)
    acctker = AccTracker()
    
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch+1,epochs))
        
        for i, (feature, label) in enumerate(train_loader):            
            
            # zero the parameter gradients
            opt.zero_grad()
            model.zero_grad()
            
            feature = feature.view(-1,2048)
            # image = image.view(-1,3,224,224)
            feature = feature.to(device)
            label = label.to(device)
          
            pred = model(feature, mode ='f')
            
            ce_loss = torch.nn.CrossEntropyLoss()
            
            label = label.to(torch.int64)
            
            loss = ce_loss(pred, label)
            
            loss.backward()
            
            opt.step()
            
            class_pred = torch.argmax(pred, dim = 1).view(label.shape)
            
            acctker.accumlate(cor_sample = torch.sum(class_pred == label).item(), num_sample = label.shape[0])            
            ltker.accumlate([loss.item()])

            if i%10 == 0:
                print('.', end = '')
            
        print("\nloss:{} / acc:{}".format(ltker.get_epoch_loss(loss_names[0]), acctker.get_epoch_acc()))

        ltker.update()
        acctker.update()

        #save checkpoint
        if (epoch+1) %10 == 0:
            record_file = record_root + '/task1_{}_model.pth.tar'
            save_checkpoint(model.state_dict(), filename='task1_{}_model.pth.tar'.format(epoch+1))
            
        if (epoch+1) == epochs:
            record_file = record_root + '/task1_model.pth.tar'
            save_checkpoint(model.state_dict(), filename = record_file)            
        
           
            his_dict = {"loss":ltker.get_loss_history(loss_names[0]), "acc":acctker.get_acc_history()}
            path = record_root + "/task_1_epoch_{}_history.json".format(epoch+1)
            SaveJson(path, his_dict)

def icl_training(task, last_classes, train_loader, epochs):
    
    # init hyper- parameter
    cfg = Config()
    record_root = cfg.record_root 
    feature_dim = cfg.feature_dim
    
    # ce_w = cfg.icl_loss_ratio
    epochs = cfg.icl_cnn_epoch
    lr_dict = cfg.icl_cnn_lr_dict
    
    # ss set inital
    insertion_check_list = cfg.insert_check_list    
    kernel_sizes = cfg.ss_sizes
    
    # init trainig classes
    increment_classes = cfg.incre_classes    
    target_classes = cfg.base_classes + (task - 1) * increment_classes
    origin_classes = target_classes - increment_classes

    ce_w = cfg.icl_loss_ratio

    curr_task = task
    last_task = task-1
    
    if task == 2:
        task_before_last_classes = 0
    else:
        task_before_last_classes = origin_classes - increment_classes     

    previous_epoch = 0
    previous_epoch_model_path = record_root + '//' +'task{}_{}_model.pth.tar'.format(task, previous_epoch)
    previous_epoch_kernel_path = record_root + '//'+'task{}_{}_kernel.pth.tar'.format(task, previous_epoch)    
    
    if os.path.exists(previous_epoch_model_path) and os.path.exists(previous_epoch_kernel_path):    
        
        #load current task previous epoch model and kernel
        cnn = Load_Model(target_classes, curr_task, previous_epoch)
        kernel_list2 = Load_Kernel(insertion_check_list, kernel_sizes, curr_task, training = True, previous_epoch = previous_epoch)
        
        #load last task kernel fixed kernel
        kernel_list1 = Load_Kernel(insertion_check_list, kernel_sizes, last_task)
        
    else:   
        previous_epoch = 0
        
        # load last task model
        cnn1 = Load_Model(origin_classes, last_task)
        cnn2 = Load_Model(origin_classes, last_task)        
        cnn2 = model_reconstruction(cnn2, origin_classes, target_classes)
        # print(cnn1.fc2.weight.data.shape)
        # print(cnn2.fc2.weight.data.shape)
        # sys.exit()
   
    model0 = cnn1.to(device)
    model1 = cnn2.to(device)
    
    model0.eval()
    model1.FE.eval()

    # fix last layer
    # para_dict  = [ {'params': model1.fc2.parameters()}]    

    #NO fixed
    para_dict  = [ {'params': model1.fc1.parameters()},
                   {'params': model1.fc2.parameters()}]                     
                   


    lr = lr_dict["base"]# 0.0009 #0.005
    opt = optim.Adam(para_dict, lr, weight_decay = 0.0005, amsgrad = True)     

    # epochs = 100
    loss_names = ["total", "ce", "kl"] 
    ltker = LossTracker(num_loss = 3 , loss_names = loss_names)
    acctker = AccTracker()
    
    # initial weight keeper for fixing part weight of specific layer
    # load last fc layer old weight
    wker = WeightKeeper(model1.fc2, interval=[0,origin_classes])   
    
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch+1,epochs))
        
        for i, (feature, label) in enumerate(train_loader): 
            # sys.exit()
            
            if i%5 == 0:
                print('.', end = '')        
            
            # zero the parameter gradients
            opt.zero_grad()
            model0.zero_grad()
            model1.zero_grad()
            
            #forward
            feature = feature.view(-1,2048)
            feature = feature.to(device)
            label = label.to(device)

            # output_list = model(feature, inp_mode = "f")
                     
            pred0 =  model0(feature, mode = "f")
            pred1 =  model1(feature, mode = "f")
            
            # loss1 for new class clf
            ce_loss = torch.nn.CrossEntropyLoss()
            
            label = label.to(torch.int64)            
            
            loss1 = ce_loss(pred1, label) 
        
            # loss 2 for new / old balance
            # loss2 = torch.zeros(1).to(device)
            # loss2 = KLDivLoss(pred0, pred1, origin_classes, init_class = task_before_last_classes)
            # loss2 = KLDivLoss(pred0, pred1, origin_classes, init_class = 0)
            loss2 = KDLoss(pred0, pred1, origin_classes)

            # ''' total loss '''
            loss = ce_w[task-2] * loss1 + (1-ce_w[task-2] ) * loss2
            # loss = loss1 + loss2
            # loss = 0.3 * loss1 + 0.7*loss2
        
            loss.backward()                        

            opt.step()  
            
            #cover fc3 weight and bias
            # model1.fc2 = wker.recover(model1.fc2)   
            
            class_pred = torch.argmax(pred1, dim = 1).view(label.shape)
           
            acctker.accumlate(torch.sum(class_pred == label).item(), label.shape[0])
            ltker.accumlate([loss.item(), loss1.item(), loss2.item()])            
        
        
        print("\nloss:{} / acc:{}".format(ltker.get_epoch_loss(loss_names[0]), acctker.get_epoch_acc()))
        print("ce_loss:{} / kl_loss:{} ".format(ltker.get_epoch_loss(loss_names[1]), ltker.get_epoch_loss(loss_names[2])))        

        ltker.update()
        acctker.update()
                

        # if (epoch+1) %20 == 0:        
            
            # epoch_r =str(previous_epoch + epoch + 1)
            # file = 'task{}_{}_model.pth.tar'.format(task,epoch_r)
            # save_checkpoint(model.basic_model.state_dict(), filename = file)

            # file = 'task{}_{}_kernel.pth.tar'.format(task,epoch_r)
            # save_checkpoint(model.kernel_set2.state_dict(), filename = file)
        
        if (epoch+1)== epochs:         
            
            #save model
            file = record_root+"//"+'task{}_model.pth.tar'.format(task)
            save_checkpoint(model1.state_dict(), filename = file)

            # file = record_root+"//"+'task{}_kernel.pth.tar'.format(task)
            # save_checkpoint(model.kernel_set2.state_dict(), filename = file)    

            file = record_root+"//"+'task{}_{}_opt.pth.tar'.format(task, epoch + 1)
            opt_dict = opt.state_dict()  
            save_checkpoint(opt_dict, filename = file)
         
            #record from last checkpoint
            if previous_epoch !=0:
                json_path =  record_root+"//"+'task_{}_epoch_{}_history.json'.format(task, previous_epoch)
                with open(json_path) as f:
                    data = json.load(f)
                loss_history = data["loss"] + get_loss_history(loss_names[0])
                acc_history =  data["acc"] + acctker.get_acc_history()
                loss1_history = data["loss1"] + get_loss_history(loss_names[1])
                loss2_history = data["loss2"] + get_loss_history(loss_names[2])

            #Save loss & ACC Record
            his_dict = {"loss":ltker.get_loss_history(loss_names[0]), 
                        "acc":acctker.get_acc_history(),
                        "loss1":ltker.get_loss_history(loss_names[1]),
                        "loss2":ltker.get_loss_history(loss_names[2])}
                        
            path =  record_root+"//"+"task_{}_epoch_{}_history.json".format(task, previous_epoch + epoch+1)

            SaveJson(path, his_dict)            

