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
from Record import *
from Track import LossTracker, AccTracker
from Config import *

device = "cuda" if torch.cuda.is_available() else "cpu"

cfg = Config()
record_root = cfg.record_root 

if os.path.exists(record_root) ==False:
    os.mkdir(record_root)

def train_base_CNN(train_loader, epochs = 30):

    # hyper-parameter
    cfg = Config()
    record_root = cfg.record_root        
    feature_dim = cfg.feature_dim
    
    # init model
    model = CNN(cfg.base_classes)
    model = model.to(device) 
    model.train()    
    model.FE.eval()
    
    # init hyper-parameter
    epochs = cfg.base_cnn_epochs
    lr = cfg.base_cnn_lr
    wd = cfg.base_cnn_wd

    para_dict = [{'params': model.parameters()}]
    opt = optim.Adam(para_dict, lr, weight_decay = wd, amsgrad = True)
    
    loss_history = []
    loss1_history = []
    
    loss_names = ["ce_loss"] # cross-entropy
    ltker = LossTracker(num_loss = 1 , loss_names = loss_names)
    acctker = AccTracker()

    # training loop
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

def train_icl_CNN(task, train_loader, epochs=100):
   
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


def AE_training(train_loader, epochs):

    VAEModel = VAE()
    VAEModel = VAEModel.to(device) 
    VAEModel.train()  
    task = 1
    print_epoch = epochs   
    lr = cfg.base_lr
    
    para_dict = [{'params': VAEModel.parameters()}]

    # opt = optim.Adam(para_dict, lr, weight_decay =  0.0005, amsgrad = True)
    opt = optim.Adam(para_dict, lr)
     
    loss_names = ["total","mse","kl"] 
    ltker = LossTracker(num_loss = 3 , loss_names = loss_names)
    # RatioFTer = LossRatioFinetune()
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch+1,epochs))        
        # print("\nlr:{}".format(RatioFTer.get_beta()))
        
        for i, (data, label) in enumerate(train_loader):            
            
            # zero the parameter gradients
            opt.zero_grad()
            VAEModel.zero_grad()
            
            data = torch.flatten(data)
            data = data.view(-1,784)
            data = data.to(device)
            label = label.to(device)

            # forward           
            mean, var, z_code, x_head = VAEModel(data)
          
            # vae loss
            vae_loss1 = ReconLoss(data , x_head) 
            vae_loss2 = VAE_KLDivLoss(mean, var)

            
            # loss = vae_loss1 + RatioFTer.get_beta() * vae_loss2
            loss = vae_loss1 +  vae_loss2

            loss.backward()
            
            opt.step()
            
            
            ltker.accumlate([loss.item(), vae_loss1.item(), vae_loss2.item()]) 
            
            
            batch_count = i+1
            
            if i%10 == 0:
                print('.', end = '')
           
        # print("\nloss:{} / acc:{}".format(total_loss/batch_count, acc_sample/total_sample))
        print("\nloss:{}".format(ltker.get_epoch_loss(loss_names[0])))
        print("vae mse loss : {} / vae kl loss : {}".format(ltker.get_epoch_loss(loss_names[1]), ltker.get_epoch_loss(loss_names[2])))
        
        # RatioFTer.update_beta(epoch)
        ltker.update()

        # #save checkpoint
        # if (epoch+1) %10 == 0:
            # record_file = record_root + '/task1_{}_vae_model.pth.tar'
            # save_checkpoint(model.state_dict(), filename='task1_{}_vae_model.pth.tar'.format(epoch+1))
            
        if (epoch+1) == epochs:
            # record_file = record_root + '/task1_model.pth.tar'
            # save_checkpoint(model.state_dict(), filename = record_file)      

            record_file = record_root + '/task{}_vae_model.pth.tar'.format(task)
            save_checkpoint(VAEModel.state_dict(), filename = record_file)              
        
        if (epoch+1)== epochs:                
                
            path = record_root + "/task_{}_epoch_{}_history.json".format(task, print_epoch)
            
            record = {}
            record.update({"vae_total_loss":ltker.get_loss_history(loss_names[0])})
            record.update({"vae_loss1":ltker.get_loss_history(loss_names[1])})
            record.update({"vae_loss2":ltker.get_loss_history(loss_names[2])})
              
            SaveJson(path, record)

def train_autoencoder(VAEModel, task, train_loader, epochs):

    # VAEModel = VAE
    VAEModel = VAEModel.to(device) 
    VAEModel.train() 

    icl_epoch = 100    
    

    epochs = 30 #30    
    lr = 0.001
    
    para_dict = [{'params': VAEModel.parameters()}]

    opt = optim.Adam(para_dict, lr, weight_decay =  0.0005, amsgrad = True)
    
    loss_history = []
    loss1_history = []
    vae_loss1_history = []
    vae_loss2_history = []
    
    acc_history = []
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch+1,epochs))
        acc_sample = 0
        total_sample = 0
        total_loss = 0

        vae_total_loss1 = 0
        vae_total_loss2 = 0

        for i, (feature, label) in enumerate(train_loader):            
            
            # zero the parameter gradients
            opt.zero_grad()
            VAEModel.zero_grad()
            
            feature = feature.view(-1,2048)
            feature = feature.to(device)
            label = label.to(device)

            # forward
           
            mean, var, z_code, x_head = VAEModel(feature)

          
            # vae loss
            vae_loss1 = MSELoss(feature , x_head) 
            vae_loss2 = VAE_KLDivLoss(mean, var)

            
            loss = vae_loss1 + 0.2 * vae_loss2

            loss.backward()
            
            opt.step()
                        
            total_loss += loss.item()
            vae_total_loss1 += vae_loss1.item()
            vae_total_loss2 += vae_loss2.item()
            
            
            batch_count = i+1
            
            if i%10 == 0:
                print('.', end = '')
            
        # print("\nloss:{} / acc:{}".format(total_loss/batch_count, acc_sample/total_sample))
        print("\nloss:{}".format(total_loss/batch_count))
        print("vae mse loss : {} / vae kl loss : {}".format(vae_total_loss1/batch_count, vae_total_loss2/batch_count))

        loss_history.append(total_loss/batch_count)
        vae_loss1_history.append(vae_total_loss1/batch_count)
        vae_loss2_history.append(vae_total_loss2/batch_count)

        #save checkpoint
        # if (epoch+1) %10 == 0:
            # record_file = record_root + '/task1_{}_model.pth.tar'
            # save_checkpoint(model.state_dict(), filename='task1_{}_model.pth.tar'.format(epoch+1))
            
        if (epoch+1) == epochs:
            # record_file = record_root + '/task1_model.pth.tar'
            # save_checkpoint(model.state_dict(), filename = record_file)      

            record_file = record_root + '/task{}_vae_model.pth.tar'.format(task)
            save_checkpoint(VAEModel.state_dict(), filename = record_file)              
        
        if (epoch+1)== epochs:                
            # his_dict = {"loss":loss_history, "acc":acc_history}
            if task == 1:
                print_epoch = 15
            else:
                print_epoch = icl_epoch
                
            path = record_root + "/task_{}_epoch_{}_history.json".format(task, print_epoch)
            
            with open(path) as jsonFile:
                record = json.load(jsonFile)  
                record.update({"vae_total_loss":loss_history})
                record.update({"vae_loss1":vae_loss1_history})
                record.update({"vae_loss2":vae_loss2_history})
                # his_dict = {"loss":loss_history, "acc":acc_history, "loss1": loss1_history, "vae_loss1":vae_loss1_history, "vae_loss2":vae_loss2_history}
            
            jsonFile = open(path, "w")
            jsonString = json.dumps(record)
            jsonFile.write(jsonString)
            jsonFile.close()

def base_training(train_loader, epochs = 10): 
    
    print("train CNN:")
    train_base_CNN(train_loader) 
    print("train CVAE:")
    train_base_CVAE(train_loader)
    
def icl_training(task, train_loader, epochs = 10):
    
    print("train CNN:")
    train_icl_CNN(task, train_loader)
    print("train CVAE:")
    train_icl_CVAE(task, train_loader)
 

 