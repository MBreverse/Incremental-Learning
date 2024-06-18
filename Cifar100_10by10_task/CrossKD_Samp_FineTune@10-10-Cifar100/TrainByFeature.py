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
import copy
import math

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


def train_base_CNN(model, train_loader): 
   
    model = model.to(device) 
    model.train()    
    model.FE.eval()  
    
    cfg = Config()
    epochs = cfg.base_cnn_epochs  
    lr = cfg.base_cnn_lr
         
    para_dict = [{'params': model.fc1.parameters()},{'params': model.fc2.parameters()}]

    opt = optim.Adam(para_dict, lr, weight_decay =  0.00005, amsgrad = True)
    
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
          
            pred = model(feature, "f")
            
            ce_loss = torch.nn.CrossEntropyLoss()
            
            label = label.to(torch.int64)
            
            loss = ce_loss(pred, label)
            # bce_loss = nn.BCEWithLogitsLoss()            
            # onehot = one_hot(label, cfg.base_classes)
            # loss = bce_loss(pred, onehot)            
            
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

def train_icl_CNN(task, train_loader, pse_loader):

    # init hyper- parameter
    cfg = Config()
    record_root = cfg.record_root 
    feature_dim = cfg.feature_dim
    
    # ce_w = cfg.icl_loss_ratio
    epochs = cfg.icl_cnn_epoch
    lr_dict = cfg.icl_cnn_lr_dict
    loss_ratio = cfg.loss_ratio
    
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
    total_digit = origin_classes
    cnn = Load_Model2(model_type = "re-CNN", class_digits = total_digit, task = last_task, root = cfg.record_root, name = cfg.save_cnn_name)
    cnn.fc2 = module_reconstruction(cnn.fc2, origin_classes, target_classes)
    cnn.fc2.update_init()

    # assert False
 
    # model = cnn
    # model.FE.eval()
    model = SS_Model3(cnn)
    model.build_SS()
    
    # last_ss_path = cfg.record_root + '/task{}_{}.pth.tar'.format(last_task, cfg.save_ss_name)
    # if not os.path.exists(last_ss_path):
        # model.build_SS()
    # else:
        # ss_state_dict = torch.load(last_ss_path)
        # model.build_SS()
        # model.load_SS(ss_state_dict)

    model.get_submodule("base_model").FE.eval()
    model.get_submodule("base_model").fc1.eval()
    model.get_submodule("base_model").fc2.eval()
    model = model.to(device)
    
    trainable_module = model.get_submodule("ss_manager").get_submodule("multi_SS")
    # trainable_module.eval()
    # model.get_submodule("base_model").fc2.train()
    # trainable_module = model.get_submodule("fc2")#get_submodule("ss_manager")#.get_submodule("multi_SS")
    
    # decide the training parameter
    para_dict  = [  
                    {'params': trainable_module.parameters(), "lr" : lr_dict["ss"]},
                    {'params': model.get_submodule("base_model").fc2.parameters(), "lr" : lr_dict["ss"]} ] 

    opt = optim.Adam(para_dict, weight_decay = 0.0005, amsgrad = True) 
    
    # epochs = 100
    loss_names = ["total", "ce", "cos"] 
    ltker = LossTracker(num_loss = 3 , loss_names = loss_names)
    acctker = AccTracker()
    
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch+1,epochs))
        
        for i, ((feature, label), (encoding_x, encoding_y)) in enumerate( zip(train_loader, pse_loader) ): 
            
            print('.' if i%10 == 0 else "", end = '')   
            
            # zero the parameter gradients
            opt.zero_grad()
            model.zero_grad()
            
            #forward
            feature = feature.view(-1,feature_dim)
            feature = feature.to(device)
            label = label.to(device)
            
            encoding_x = encoding_x.to(device)
            padding = torch.zeros([encoding_x.size(0), incre_classes]).to(device)
            encoding_x = torch.cat([encoding_x, padding], dim = 1)
            encoding_y = encoding_y.to(device)

            pred1 = model(feature, "f")
            pred1 = pred1#*(torch.Tensor(range(0,100)) < 20 ).to(device)

            # output_list = model(feature, inp_mode = "f")

            # pred0 = output_list[0]
            # pred1 = output_list[1]

            # loss1 for new class clf
            
            # ce_loss = torch.nn.CrossEntropyLoss()
            # label = label.to(torch.int64)                        
            # loss1 = ce_loss(pred1, label)             
            
            bce_loss = nn.BCELoss()
            sigmoid = nn.Sigmoid()
            pred_logit = sigmoid(pred1) 
            
            # mix_pred = torch.cat([pred_logit, encoding_x], 0)
            # mix_label = torch.cat([label, encoding_y], 0)

            onehot = one_hot(label, target_classes)
            # print(mix_label.unique())
            # print(onehot.shape)
            # assert False            
            loss1 = bce_loss(pred_logit, onehot)

            # loss 2 for new / old balance : Knowledge Distillation by KL divergence
            # loss2 = KLDivLoss(pred0, pred1, origin_classes, init_class = 0)
            # loss2 = KDLoss(pred0, pred1, origin_classes)
            # loss2 = torch.zeros([1]).to(device)

            # encoding_y = one_hot(encoding_y, target_classes)
            # loss2 = bce_loss(encoding_x, encoding_y)
            
            cos_Loss = nn.CosineSimilarity(dim = 1)      
            # sigmoid = nn.Sigmoid()
            # pred_logit = sigmoid(pred1)[:, 0 : origin_classes]
            loss2 = 1 + cos_Loss(torch.mean(pred_logit, dim = 0, keepdim = True), torch.mean(encoding_x, dim = 0, keepdim = True))
            
            # encoding_y = encoding_y.to(torch.int64)            
            # loss2 = KLDivLoss_NoSoftmax(encoding_x, pred1, target_classes)
            # loss2 = KLDivLoss_Sigmoid(encoding_x, pred1, target_classes)

            ''' total loss '''
            # loss = combine_loss_with_ratio(loss1, loss2, ratio = loss_ratio[task-2])
            loss = loss1 + loss2
        
            loss.backward()                        
            opt.step()  
            
            # print( model.get_submodule("base_model"))
            model.get_submodule("base_model").fc2.recover(1, [0, origin_classes])
            # print( model.get_submodule("base_model").fc2.weight.data[0,0])
            # print(cnn.fc2.weight.data[0,0:10])

            # model.get_submodule("ss_manager").get_submodule("multi_SS")[-1].alpha2.data.fill_(1.)
            # model.get_submodule("ss_manager").get_submodule("multi_SS")[-1].beta2.data.fill_(0.)

            # model pred result
            class_pred = torch.argmax(pred1, dim = 1).view(label.shape) 
            acctker.accumlate(torch.sum(class_pred == label).item(), label.shape[0])
            ltker.accumlate([loss.item(), loss1.item(), loss2.item()])       
        
        print("\nloss:{} / acc:{}".format(ltker.get_epoch_loss(loss_names[0]), acctker.get_epoch_acc()))
        print("ce_loss:{} / kD_loss:{} ".format(ltker.get_epoch_loss(loss_names[1]), ltker.get_epoch_loss(loss_names[2])))        

        ltker.update()
        acctker.update()
        
        if (epoch+1)== epochs:         
            
            # save model and ss set
            file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_cnn_name)
            save_checkpoint(model.get_submodule('base_model').state_dict(), filename = file)

            file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_ss_name)
            save_checkpoint(model.get_SS_dict(), filename = file)    

            # Save loss & ACC Record
            his_dict = {"loss":ltker.get_loss_history(loss_names[0]), 
                        "acc":acctker.get_acc_history(),
                        "loss1":ltker.get_loss_history(loss_names[1]),
                        "loss2":ltker.get_loss_history(loss_names[2])}

            path =  record_root+"//"+"task{}_{}_history.json".format(task, cfg.icl_cnn_train_his_name)
            SaveJson(path, his_dict)

def train_base_CVAE(train_loader):
    
    feat_dim = cfg.base_classes
    model =  CVAE(feat_dim, cfg.z_dim, cfg.base_classes).to(device)   
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
            data = data.view(-1, cfg.base_classes)
            # print(data.shape)
            # assert False

            onehot = one_hot(labels, cfg.base_classes)
            # print(onehot.shape)
            # assert False

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

def train_icl_CNN2(task, train_loader):

    # init hyper- parameter
    cfg = Config()
    record_root = cfg.record_root 
    feature_dim = cfg.feature_dim
    
    # ce_w = cfg.icl_loss_ratio
    epochs = cfg.icl_cnn_epoch
    lr_dict = cfg.icl_cnn_lr_dict
    loss_ratio = cfg.loss_ratio
    
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
    total_digit = origin_classes
    cnn = Load_Model2(model_type = "re-CNN", class_digits = total_digit, task = last_task, root = cfg.record_root, name = cfg.save_cnn_name)
    cnn.fc2 = module_reconstruction(cnn.fc2, origin_classes, target_classes)
    cnn.fc2.update_init()

    # assert False
 
    # model = cnn
    # model.FE.eval()
    model = SS_Model3(cnn)
    model.build_SS()
    
    # last_ss_path = cfg.record_root + '/task{}_{}.pth.tar'.format(last_task, cfg.save_ss_name)
    # if not os.path.exists(last_ss_path):
        # model.build_SS()
    # else:
        # ss_state_dict = torch.load(last_ss_path)
        # model.build_SS()
        # model.load_SS(ss_state_dict)
        
    import copy
    
    model1 =  copy.deepcopy(model.get_submodule("base_model")).to(device)
    
    model.get_submodule("base_model").FE.eval()
    # model.get_submodule("base_model").fc1.eval()
    # model.get_submodule("base_model").fc2.eval()
    model = model.to(device)
    
    trainable_module = model.get_submodule("ss_manager").get_submodule("multi_SS")
    # trainable_module.eval()
    # model.get_submodule("base_model").fc2.train()
    # trainable_module = model.get_submodule("fc2")#get_submodule("ss_manager")#.get_submodule("multi_SS")
    
    # decide the training parameter
    para_dict  = [  
                    {'params': trainable_module.parameters(), "lr" : lr_dict["ss"]},
                    # {'params': model.get_submodule("base_model").fc1.parameters(), "lr" : lr_dict["ss"]},
                    {'params': model.get_submodule("base_model").fc2.parameters(), "lr" : lr_dict["ss"]} ] 

    opt = optim.Adam(para_dict, weight_decay = 0.0005, amsgrad = True) 
    
    # epochs = 100
    loss_names = ["total", "ce", "cos"] 
    ltker = LossTracker(num_loss = 3 , loss_names = loss_names)
    acctker = AccTracker()
    
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
            

            pred1 = model(feature, "f")
            pred0 = model1(feature, "f")

            bce_loss = nn.BCELoss()
            sigmoid = nn.Sigmoid()
            pred_logit = sigmoid(pred1) 
            
            onehot = one_hot(label, target_classes)        
            loss1 = bce_loss(pred_logit, onehot)

            # loss 2 for new / old balance : Knowledge Distillation by KL divergence
            # loss2 = KLDivLoss(pred0, pred1, origin_classes, init_class = 0)
            loss2 = KD_Sigmoid_Loss(pred0, pred1, origin_classes)
            # loss2 = torch.zeros([1]).to(device)

            # encoding_y = one_hot(encoding_y, target_classes)
            # loss2 = bce_loss(encoding_x, encoding_y)


            ''' total loss '''
            # loss = combine_loss_with_ratio(loss1, loss2, ratio = loss_ratio[task-2])
            loss = loss1 + loss2
        
            loss.backward()                        
            opt.step()  
            
            # print( model.get_submodule("base_model"))
            model.get_submodule("base_model").fc2.recover(1, [0, origin_classes])


            # model pred result
            class_pred = torch.argmax(pred1, dim = 1).view(label.shape) 
            acctker.accumlate(torch.sum(class_pred == label).item(), label.shape[0])
            ltker.accumlate([loss.item(), loss1.item(), loss2.item()])       
        
        print("\nloss:{} / acc:{}".format(ltker.get_epoch_loss(loss_names[0]), acctker.get_epoch_acc()))
        print("ce_loss:{} / kD_loss:{} ".format(ltker.get_epoch_loss(loss_names[1]), ltker.get_epoch_loss(loss_names[2])))        

        ltker.update()
        acctker.update()
        
        if (epoch+1)== epochs:         
            
            # save model and ss set
            file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_cnn_name)
            save_checkpoint(model.get_submodule('base_model').state_dict(), filename = file)

            file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_ss_name)
            save_checkpoint(model.get_SS_dict(), filename = file)    

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


def train_icl_CNN_cross_distill(task, train_loader):

    # init hyper- parameter
    cfg = Config()
    record_root = cfg.record_root 
    feature_dim = cfg.feature_dim
    
    # ce_w = cfg.icl_loss_ratio
    epochs = cfg.icl_cnn_epoch
    lr_dict = cfg.icl_cnn_lr_dict
    loss_ratio = cfg.loss_ratio
    
    # ss set inital
    insertion_check_list = cfg.insert_check_list    
    kernel_sizes = cfg.ss_sizes
    
    # init trainig classes
    base_classes = cfg.base_classes
    incre_classes = cfg.incre_classes    
    target_classes = cfg.base_classes + (task - 1) * incre_classes
    origin_classes = target_classes - incre_classes
    
    # init training task
    curr_task = task
    last_task = task-1

    # load last task model
    total_digit = origin_classes
    cnn = Load_Model2(model_type = "re-CNN", class_digits = total_digit, task = last_task, root = cfg.record_root, name = cfg.save_cnn_name)
    
    old_model = copy.deepcopy(cnn).to(device)
    
    cnn.fc2 = module_reconstruction(cnn.fc2, origin_classes, target_classes)
    cnn.fc2.update_init()
    new_model = cnn

    old_model.eval()
    new_model.FE.eval()
    
    old_model = old_model.to(device)
    new_model = new_model.to(device)
    
    trainable_module = new_model.get_submodule("fc2")
    
    # decide the training parameter
    para_dict  = [  
                    {'params': trainable_module.parameters(), "lr" : lr_dict["base"]}]
                    # {'params': model.get_submodule("base_model").fc1.parameters(), "lr" : lr_dict["ss"]},
                    # {'params': model.get_submodule("base_model").fc2.parameters(), "lr" : lr_dict["ss"]} ] 

    opt = optim.Adam(para_dict, weight_decay = 0.00005, amsgrad = True) 
    
    
    loss_names = ["total", "ce", "kd"] 
    ltker = LossTracker(num_loss = 3 , loss_names = loss_names)
    acctker = AccTracker()
    
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch+1,epochs))
        
        if (epoch+1)%10==0:
            origin_dict = opt.state_dict()

            for param in origin_dict['param_groups']:
                print()
                origin_lr = param['lr']
                param.update({"lr":origin_lr*0.1})
            opt.load_state_dict(origin_dict)          
        
        for i, (feature, label) in enumerate(train_loader): 
            
            print('.' if i%10 == 0 else "", end = '')   
            
            # zero the parameter gradients
            opt.zero_grad()
            new_model.zero_grad()
            
            #forward
            feature = feature.view(-1,feature_dim)
            feature = feature.to(device)
            label = label.to(device)
            

            pred1 = new_model(feature, "f")
            pred0 = old_model(feature, "f")
            
            ce_loss = torch.nn.CrossEntropyLoss()            
            label = label.to(torch.int64)                        
            loss1 = ce_loss(pred1, label)             
            
            loss2 = CrossKDLoss(pred0, pred1, base_classes, incre_classes, origin_classes)


            ''' total loss '''
            # loss = combine_loss_with_ratio(loss1, loss2, ratio = loss_ratio[task-2])
            loss = loss1 + loss2
        
            loss.backward()                        
            opt.step()  
            
            # print( model.get_submodule("base_model"))
            # model.get_submodule("base_model").fc2.recover(1, [0, origin_classes])


            # model pred result
            class_pred = torch.argmax(pred1, dim = 1).view(label.shape) 
            acctker.accumlate(torch.sum(class_pred == label).item(), label.shape[0])
            ltker.accumlate([loss.item(), loss1.item(), loss2.item()])       
        
        print("\nloss:{} / acc:{}".format(ltker.get_epoch_loss(loss_names[0]), acctker.get_epoch_acc()))
        print("ce_loss:{} / kD_loss:{} ".format(ltker.get_epoch_loss(loss_names[1]), ltker.get_epoch_loss(loss_names[2])))        

        ltker.update()
        acctker.update()
        
        if (epoch+1)== epochs:         
            
            # save model and ss set
            file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_cnn_name)
            save_checkpoint(new_model.state_dict(), filename = file)

            # file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_ss_name)
            # save_checkpoint(model.get_SS_dict(), filename = file)    

            # Save loss & ACC Record
            his_dict = {"loss":ltker.get_loss_history(loss_names[0]), 
                        "acc":acctker.get_acc_history(),
                        "loss1":ltker.get_loss_history(loss_names[1]),
                        "loss2":ltker.get_loss_history(loss_names[2])}

            path =  record_root+"//"+"task{}_{}_history.json".format(task, cfg.icl_cnn_train_his_name)
            SaveJson(path, his_dict)


def icl_balance_finetune(task, train_loader):

    # init hyper- parameter
    cfg = Config()
    record_root = cfg.record_root 
    feature_dim = cfg.feature_dim
    
    # ce_w = cfg.icl_loss_ratio
    epochs = cfg.icl_finetune_epoch
    lr_dict = cfg.icl_cnn_lr_dict
    loss_ratio = cfg.loss_ratio
    
    # ss set inital
    insertion_check_list = cfg.insert_check_list    
    kernel_sizes = cfg.ss_sizes
    
    # init trainig classes
    base_classes = cfg.base_classes
    incre_classes = cfg.incre_classes    
    target_classes = cfg.base_classes + (task - 1) * incre_classes
    origin_classes = target_classes - incre_classes
    
    # init training task
    curr_task = task
    last_task = task-1

    # load last task model
    total_digit = target_classes
    cnn = Load_Model2(model_type = "re-CNN", class_digits = total_digit, task = curr_task, root = cfg.record_root, name = cfg.save_cnn_name)
    
    old_model = copy.deepcopy(cnn).to(device)
    
    # cnn.fc2 = module_reconstruction(cnn.fc2, origin_classes, target_classes)
    cnn.fc2.update_init()
    new_model = cnn

    old_model.eval()
    new_model.FE.eval()
    
    old_model = old_model.to(device)
    new_model = new_model.to(device)
    
    trainable_module = new_model.get_submodule("fc2")
    
    # decide the training parameter
    para_dict  = [  
                    {'params': trainable_module.parameters(), "lr" :  cfg.icl_finetune_lr}]
                    # {'params': model.get_submodule("base_model").fc1.parameters(), "lr" : lr_dict["ss"]},
                    # {'params': model.get_submodule("base_model").fc2.parameters(), "lr" : lr_dict["ss"]} ] 

    opt = optim.Adam(para_dict, weight_decay = 0.0005, amsgrad = True) 
    
    
    loss_names = ["total", "ce", "kd"] 
    ltker = LossTracker(num_loss = 3 , loss_names = loss_names)
    acctker = AccTracker()
    
    for epoch in range(epochs):
        print("epoch : {}/{}".format(epoch+1,epochs))
        
        if (epoch+1)%10==0:
            origin_dict = opt.state_dict()

            for param in origin_dict['param_groups']:
                origin_lr = param['lr']
                param.update({"lr":origin_lr*0.1})
            opt.load_state_dict(origin_dict)            
            
        
        for i, (feature, label) in enumerate(train_loader): 
            
            print('.' if i%10 == 0 else "", end = '')   
            
            # zero the parameter gradients
            opt.zero_grad()            
            new_model.zero_grad()
            
            #forward
            feature = feature.view(-1,feature_dim)
            feature = feature.to(device)
            label = label.to(device)
            

            pred1 = new_model(feature, "f")
            pred0 = old_model(feature, "f")
            
            ce_loss = torch.nn.CrossEntropyLoss()            
            label = label.to(torch.int64)                        
            loss1 = ce_loss(pred1, label)             
            
            # loss2 = CrossKDLoss(pred0, pred1, base_classes, incre_classes, origin_classes)
            # loss2 = GeneralKDLoss(pred0, pred1, origin_classes, target_classes-1)
            loss2 = torch.zeros([1]).to(device)


            ''' total loss '''
            # loss = combine_loss_with_ratio(loss1, loss2, ratio = loss_ratio[task-2])
            loss = loss1 + loss2
        
            loss.backward()                        
            opt.step()  
            
            # print( model.get_submodule("base_model"))
            # model.get_submodule("base_model").fc2.recover(1, [0, origin_classes])


            # model pred result
            class_pred = torch.argmax(pred1, dim = 1).view(label.shape) 
            acctker.accumlate(torch.sum(class_pred == label).item(), label.shape[0])
            ltker.accumlate([loss.item(), loss1.item(), loss2.item()])       
        
        print("\nloss:{} / acc:{}".format(ltker.get_epoch_loss(loss_names[0]), acctker.get_epoch_acc()))
        print("ce_loss:{} / kD_loss:{} ".format(ltker.get_epoch_loss(loss_names[1]), ltker.get_epoch_loss(loss_names[2])))        

        ltker.update()
        acctker.update()
        
        if (epoch+1)== epochs:         
            
            # save model and ss set
            file = record_root+"//"+'task{}_{}.pth.tar'.format(task, cfg.save_cnn_name)
            save_checkpoint(new_model.state_dict(), filename = file)


            # Save loss & ACC Record
            his_dict = {"loss":ltker.get_loss_history(loss_names[0]), 
                        "acc":acctker.get_acc_history(),
                        "loss1":ltker.get_loss_history(loss_names[1]),
                        "loss2":ltker.get_loss_history(loss_names[2])}

            path =  record_root+"//"+"task{}_{}_history.json".format(task, cfg.icl_finetune_his_name)
            SaveJson(path, his_dict)

    
    

def base_training(model, train_loader):
    
    train_base_CNN(model, train_loader)
    
    # encode_set = CNN_Encoding(1, train_loader)
    # cvae_train_loader = DataLoader(encode_set, batch_size = cfg.batch_size, shuffle = True)    
    # train_base_CVAE(cvae_train_loader)

def icl_training(task, old_train_set , new_train_set):
    
    # pse_set = CVAE_PseudoExempar(sample_per_class = 500, class_digits = [0,9], task = task-1)
    # pse_loader = DataLoader(pse_set, batch_size = cfg.batch_size, shuffle = True )
    
    # train_icl_CNN2(task, train_loader)
    ConcateSet = torch.utils.data.ConcatDataset([old_train_set, new_train_set])        
    train_loader = DataLoader(ConcateSet, batch_size = cfg.batch_size, shuffle = True)     
    train_icl_CNN_cross_distill(task, train_loader)
    
    prev_classes = cfg.base_classes + cfg.incre_classes * (task-1)
    
    # fix_memory = cfg.fixed_memory
    
    num_exemplar = cfg.num_exemplar
    new_exemplar_set = Exemplar_Feature_Set(dataset = new_train_set, num_exemplar = num_exemplar, mode = "rand", label_format = "origin", )

    BalanceConcateSet = torch.utils.data.ConcatDataset([old_train_set, new_exemplar_set])        
    train_loader = DataLoader(BalanceConcateSet, batch_size = cfg.batch_size, shuffle = True)            
    icl_balance_finetune(task, train_loader)
    # assert False
    
    
    

    
    