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

if os.path.exists(record_root) ==False:
    os.mkdir(record_root)

CUDA_LAUNCH_BLOCKING = 1

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
          
            # pred = model(image)

            resp = model.fc1(feature)
            resp = nn.functional.dropout(resp, p=0.25, training = True, inplace=False)            
            resp = nn.functional.relu(resp, inplace=False)
            
            pred = model.fc2(resp)
            # resp = nn.functional.dropout(resp, p=0.25, training = True, inplace=False)            
            # resp = nn.functional.relu(resp, inplace=False)      
            
            # output = model.fc3(resp)
            
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
        if (epoch+1) == epochs:
            
            # save model
            record_file = record_root + '/task1_{}.pth.tar'.format(cfg.save_cnn_name)
            save_checkpoint(model.state_dict(), filename = record_file)            
            
            # save metric            
            his_dict = {"loss":ltker.get_loss_history(loss_names[0]), "acc":acctker.get_acc_history()}
            path = record_root + "/task1_{}_history.json".format(cfg.base_cnn_train_his_name)
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
    incre_classes = cfg.incre_classes    
    target_classes = cfg.base_classes + (task - 1) * incre_classes
    origin_classes = target_classes - incre_classes
    
    # init training task
    curr_task = task
    last_task = task-1

    # load last task model
    cnn = Load_Model(origin_classes, last_task)   
    cnn.fc2 = module_reconstruction(cnn.fc2, origin_classes, target_classes, first = False)
    
    # load last task fixed ss set
    ss_set1 = Load_Kernel(insertion_check_list, kernel_sizes, last_task)
    
    # new  ss set for this task
    ss_set2 = Multi_Scale_Shift_Kernel(insertion_check_list, kernel_sizes)        

    # combine cnn and kernels
    model = SS_Model(cnn, insertion_check_list, kernel_sizes)
    model.kernel_set1 = ss_set1
    model.kernel_set2 = ss_set2       
    model = model.to(device)
    
    # decide the training parameter
    para_dict  = [  {'params': model.kernel_set2.parameters(), "lr" : lr_dict["ss"]},
                    {'params': model.basic_model.fc2.parameters(), "lr" : lr_dict["base"]}]      

    opt = optim.Adam(para_dict, weight_decay = 0.0005, amsgrad = True) 
    
    # epochs = 100
    loss_names = ["total", "ce", "kl"] 
    ltker = LossTracker(num_loss = 3 , loss_names = loss_names)
    acctker = AccTracker()
    
    # initial weight keeper for fixing part weight of specific layer
    # load last fc layer old weight
    wker = WeightKeeper(model.basic_model.fc2, interval=[0,origin_classes])   
    ss_init = SSInitializer(scale_idxs = [1,3], shift_idxs = [0,2,4])
    
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch+1,epochs))
        
        for i, (feature, label) in enumerate(train_loader): 
            
            print('.' if i%10 == 0 else "", end = '')   
            
            # zero the parameter gradients
            opt.zero_grad()
            model.zero_grad()
            
            #forward
            feature = feature.view(-1,feature_dim)
            feature = feature.to(device)
            label = label.to(device)

            output_list = model(feature, inp_mode = "f")
                     
            pred0 = output_list[0]
            pred1 = output_list[1]

            
            # loss1 for new class clf
            ce_loss = torch.nn.CrossEntropyLoss()
            
            label = label.to(torch.int64)            
            
            loss1 = ce_loss(pred1, label) 
            

            # loss 2 for new / old balance : Knowledge Distillation by KL divergence
            loss2 = KLDivLoss(pred0, pred1, origin_classes, init_class = 0)

            ''' total loss '''
            # loss = combine_loss_with_ratio(loss1, loss2, ratio = ce_w[task-2]):
            loss = loss1 + loss2
        
            loss.backward()                        

            opt.step()  
            
            # recover cnn last weight and bias and specify ss set 
            model.basic_model.fc2 = wker.recover(model.basic_model.fc2)    
            model.kernel_set2 = ss_init.recover(model.kernel_set2) 

            # model pred result
            class_pred = torch.argmax(pred1, dim = 1).view(label.shape)           
            acctker.accumlate(torch.sum(class_pred == label).item(), label.shape[0])
            ltker.accumlate([loss.item(), loss1.item(), loss2.item()])                    
        
        print("\nloss:{} / acc:{}".format(ltker.get_epoch_loss(loss_names[0]), acctker.get_epoch_acc()))
        print("ce_loss:{} / kl_loss:{} ".format(ltker.get_epoch_loss(loss_names[1]), ltker.get_epoch_loss(loss_names[2])))        

        ltker.update()
        acctker.update()
        
        if (epoch+1)== epochs:         
            
            # save model and ss set
            file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_cnn_name)
            save_checkpoint(model.basic_model.state_dict(), filename = file)

            file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_ss_name)
            save_checkpoint(model.kernel_set2.state_dict(), filename = file)    

            # Save loss & ACC Record
            his_dict = {"loss":ltker.get_loss_history(loss_names[0]), 
                        "acc":acctker.get_acc_history(),
                        "loss1":ltker.get_loss_history(loss_names[1]),
                        "loss2":ltker.get_loss_history(loss_names[2])}
                        
            path =  record_root+"//"+"task{}_{}_history.json".format(task, cfg.icl_cnn_train_his_name)
            SaveJson(path, his_dict)