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


def train_base_CNN(train_loader, epochs=15): 
   
    # hyper-parameter
    cfg = Config()
    record_root = cfg.record_root        
    feature_dim = cfg.feature_dim
    epochs = cfg.base_cnn_epochs    
    lr = cfg.base_cnn_lr
    base_classes = cfg.base_classes
    
    model = CNN(base_classes)
    model = model.to(device) 
    model.train()    
    model.FE.eval()  

    para_dict = [{'params': model.parameters()}]

    opt = optim.Adam(para_dict, lr, weight_decay =  0.0005, amsgrad = True)
    
    loss_names = ["ce"] 
    ltker = LossTracker(num_loss = 1 , loss_names = loss_names)
    acctker = AccTracker()
    
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch+1,epochs))

        for i, (feature, label) in enumerate(train_loader):            
            
            print('.' if i%10 == 0 else "", end = '') 
            
            # zero the parameter gradients
            opt.zero_grad()
            model.zero_grad()
            
            feature = feature.view(-1,feature_dim)
            feature = feature.to(device)
            label = label.to(device)
          
            pred = model(feature, "f")

            ce_loss = torch.nn.CrossEntropyLoss()            
            label = label.to(torch.int64)            
            loss = ce_loss(pred, label)
            
            loss.backward()
            
            opt.step()
            
            class_pred = torch.argmax(pred, dim = 1).view(label.shape)
            
            acctker.accumlate(cor_sample = torch.sum(class_pred == label).item(), num_sample = label.shape[0])            
            ltker.accumlate([loss.item()])

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

def train_base_CVAE(train_loader, epochs=100):
    
    model =  CVAE(cfg.input_dim, cfg.z_dim, cfg.base_classes).to(device)   
    model.train()    
    
    epochs = cfg.base_cvae_epochs
    lr = cfg.base_cvae_lr
    
    optimizer = optim.Adam(model.parameters(), lr = lr )
    
    loss_names = ["total", "recon", "kl"] 
    ltker = LossTracker(num_loss = 3 , loss_names = loss_names)

    for epoch in range(epochs):
        
        print("\nepoch {}/{}".format(epoch, epochs))

        for batch_idx, (data, labels) in enumerate(train_loader):
            print('.' if batch_idx%5 == 0 else "", end = '') 
        
            data, labels = data.to(device), labels.to(device)
            data = data.view(-1, cfg.input_dim)
            
            onehot = one_hot(labels, cfg.base_classes)

            recon_batch, mu, logvar = model(data, onehot)

            optimizer.zero_grad()
            
            loss1, loss2 = cvae_loss_function(recon_batch, data, mu, logvar)
            
            loss = loss1 + loss2
            
            loss.backward()
    
            optimizer.step()
            
            ltker.accumlate([loss.item(), loss1.item(), loss2.item()]) 
        
        print("loss:{} ".format(ltker.get_epoch_loss(loss_names[0]))) 
        print("Recon:{} KL:{}".format(ltker.get_epoch_loss(loss_names[1]), ltker.get_epoch_loss(loss_names[2]))) 
        
        ltker.update()

        if epoch == epochs-1:

            record_file = record_root + '/task1_{}.pth.tar'.format(cfg.save_cvae_name)
            save_checkpoint(model.state_dict(), filename = record_file)  
            
            # Save loss & ACC Record
            his_dict = {"loss":ltker.get_loss_history(loss_names[0]), 
                        "loss1":ltker.get_loss_history(loss_names[1]),
                        "loss2":ltker.get_loss_history(loss_names[2])}
                        
            path =  record_root+"//"+"task1_{}_history.json".format(cfg.base_cvae_train_his_name)
            SaveJson(path, his_dict)    

def train_icl_CNN(task, train_loader):
    
    # init hyper- parameter
    cfg = Config()
    record_root = cfg.record_root 
    feature_dim = cfg.feature_dim
    
    # ce_w = cfg.icl_loss_ratio
    epochs = cfg.icl_epoch
    lr_dict = cfg.icl_lr_dict

    # init trainig classes
    incre_classes = cfg.incre_classes    
    target_classes = cfg.base_classes + (task - 1) * incre_classes
    origin_classes = target_classes - incre_classes
    
    # init training task
    curr_task = task
    last_task = task-1

    # load last task model
    # cnn1 = Load_Model(origin_classes, last_task)  
    cnn2 = Load_Model(origin_classes, last_task)     
    cnn2.fc2 = module_reconstruction(cnn2.fc2, origin_classes, target_classes, first = False)

    model2 = cnn2.to(device)    
    model2.train()
    model2.FE.eval()
 
    # decide the training parameter
    para_dict  = [  {'params': model2.fc1.parameters(), "lr" : lr_dict["base"]},
                    {'params': model2.fc2.parameters(), "lr" : lr_dict["base"]}]      

    opt = optim.Adam(para_dict, weight_decay = 0.0005, amsgrad = True) 

    loss_names = ["ce"] 
    ltker = LossTracker(num_loss = 1 , loss_names = loss_names)
    acctker = AccTracker()
    
    # initial weight keeper for fixing part weight of specific layer
    # load last fc layer old weight
    # wker = WeightKeeper(model2.fc2, interval=[0,origin_classes])   
    # ss_init = SSInitializer(scale_idxs = [1,3], shift_idxs = [0,2,4])
    
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch+1,epochs))
        
        for i, (feature, label) in enumerate(train_loader): 
            
            print('.' if i%10 == 0 else "", end = '')   
            
            # zero the parameter gradients
            opt.zero_grad()
            model2.zero_grad()
            
            #forward
            feature = feature.view(-1,feature_dim)
            feature = feature.to(device)
            label = label.to(device)
            
            pred1 = model2(feature, "f")

            # loss1 for new class clf
            ce_loss = torch.nn.CrossEntropyLoss()
            
            label = label.to(torch.int64)            
            
            # print(pred1)
            # loss = NL(p, label)
            loss = ce_loss(pred1, label) 

            ''' total loss '''
            # loss = combine_loss_with_ratio(loss1, loss2, ratio = ce_w[task-2]):
            # loss = loss1 #+ loss2
        
            loss.backward()                        

            opt.step()  
            
            # # recover cnn last weight and bias and specify ss set 
            # model2.fc2 = wker.recover(model2.fc2)    
            # model.kernel_set2 = ss_init.recover(model.kernel_set2) 

            # model pred result
            class_pred = torch.argmax(pred1, dim = 1).view(label.shape)  
            acctker.accumlate(torch.sum(class_pred == label).item(), label.shape[0])
            # ltker.accumlate([loss.item(), loss1.item(), loss2.item()]) 
            ltker.accumlate([loss.item()])              
        
        print("\nloss:{} / acc:{}".format(ltker.get_epoch_loss(loss_names[0]), acctker.get_epoch_acc()))
        # print("ce_loss:{} / kd_loss:{} ".format(ltker.get_epoch_loss(loss_names[1]), ltker.get_epoch_loss(loss_names[2])))        

        ltker.update()
        acctker.update()
        
        if (epoch+1)== epochs:         
            
            # save model and ss set
            file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_cnn_name)
            save_checkpoint(model2.state_dict(), filename = file)

            # file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_ss_name)
            # save_checkpoint(model.kernel_set2.state_dict(), filename = file)    

            # Save loss & ACC Record
            his_dict = {"loss":ltker.get_loss_history(loss_names[0]), 
                        "acc":acctker.get_acc_history()}
                        # "loss1":ltker.get_loss_history(loss_names[1]),
                        # "loss2":ltker.get_loss_history(loss_names[2])}
                        
            path =  record_root+"//"+"task{}_{}_history.json".format(task, cfg.icl_cnn_train_his_name)
            SaveJson(path, his_dict) 

def train_icl_CVAE(task, train_loader, epochs = 30):
    
    cur_classes = cfg.base_classes + (task-1)* cfg.incre_classes
    last_classes = cur_classes - cfg.incre_classes
    
    model = CVAE(cfg.input_dim, cfg.z_dim, last_classes)
    model_dict = torch.load(cfg.record_root + "/task{}_{}.pth.tar".format(task-1, cfg.save_cvae_name))
    model.load_state_dict(model_dict)
    
    model.decoder.de1 = module_reconstruction(model.decoder.de1, last_classes + cfg.z_dim, cur_classes + cfg.z_dim, first = True)
    model.encoder.ShareFC = module_reconstruction(model.encoder.ShareFC, last_classes + cfg.input_dim, cur_classes + cfg.input_dim, first = True)
    model = model.to(device)
    
    epochs = cfg.icl_cvae_epochs
    lr = cfg.icl_cvae_lr
    
    optimizer = optim.Adam( [{'params':model.parameters()}], lr = lr)

    loss_names = ["total", "Recon", "kl"]
    ltker = LossTracker(num_loss = 2 , loss_names = loss_names)
    
    for epoch in range(epochs):
        print("\nepoch {}/{}".format(epoch, epochs))
        
        for batch_idx, (data, labels) in enumerate(train_loader):
            print('.' if batch_idx%5 == 0 else "", end = '') 
        
            data, labels = data.to(device), labels.to(device)
            data = data.view(-1, cfg.input_dim)
            labels = one_hot(labels, cur_classes).to(device)
            
            recon_batch, mu, logvar = model(data, labels)
            
            optimizer.zero_grad()
            model.zero_grad()
            
            loss1, loss2 = cvae_loss_function(recon_batch, data, mu, logvar)            
            
            loss = loss1 + loss2
            
            loss.backward()            

            optimizer.step()            
            
            ltker.accumlate([loss.item(), loss1.item(), loss2.item()]) 
        
        print("loss:{} ".format(ltker.get_epoch_loss(loss_names[0]))) 
        print("Recon:{} KL:{}".format(ltker.get_epoch_loss(loss_names[1]), ltker.get_epoch_loss(loss_names[2])))         
        
        ltker.update()
    
        if epoch == epochs-1:

            record_file = record_root + '/task{}_{}.pth.tar'.format(task, cfg.save_cvae_name)
            save_checkpoint(model.state_dict(), filename = record_file)
            
            # Save loss & ACC Record
            his_dict = {"loss":ltker.get_loss_history(loss_names[0]), 
                        "loss1":ltker.get_loss_history(loss_names[1]),
                        "loss2":ltker.get_loss_history(loss_names[2])}
                        
            path =  record_root+"//"+"task{}_{}_history.json".format(task, cfg.icl_cvae_train_his_name) 
            SaveJson(path, his_dict)             

def base_training(train_loader, epochs = 10): 
    
    print("train CNN:")
    train_base_CNN(train_loader) 
    print("train CVAE:")
    train_base_CVAE(train_loader)

def icl_training(task, train_loader, epochs = 10):
    
    print("train CNN:")
    # train_icl_CNN(task, train_loader)
    train_icl_CNN(task, train_loader)

    print("train CVAE:")
    train_icl_CVAE(task, train_loader)

