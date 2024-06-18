# Loss function
import torch
import torch.nn as nn


 # '''loss2 v1 KL loss'''
def KLDivLoss(pred0, pred1, origin_classes, init_class = 0):

    target = pred0
    soft_target = nn.functional.softmax(target[:,init_class:origin_classes], dim=1)

    predict = pred1
    soft_predict = nn.functional.softmax(predict, dim=1)[:,init_class:origin_classes]
    
    soft_predict = torch.log(soft_predict)
    
    kl_loss = nn.KLDivLoss(reduction = 'batchmean')#'batchmean')

    loss2 = kl_loss(soft_predict, soft_target)

    return loss2
    
def KLDivLoss_NoSoftmax(pred0, pred1, origin_classes, init_class = 0):

    target = pred0
    soft_target = target[:,init_class:origin_classes]

    predict = pred1
    soft_predict = nn.functional.softmax(predict, dim=1)[:,init_class:origin_classes]
    
    soft_predict = torch.log(soft_predict)
    
    kl_loss = nn.KLDivLoss(reduction = 'batchmean')#'batchmean')

    loss2 = kl_loss(soft_predict, soft_target)

    return loss2    
    
def KLDivLoss_Sigmoid(pred0, pred1, origin_classes, init_class = 0):

    target = pred0
    soft_target = nn.functional.sigmoid(target[:,init_class:origin_classes])

    predict = pred1
    soft_predict = nn.functional.sigmoid(predict)[:,init_class:origin_classes]
    
    soft_predict = torch.log(soft_predict)
    
    kl_loss = nn.KLDivLoss(reduction = 'batchmean')#'batchmean')

    loss2 = kl_loss(soft_predict, soft_target)

    return loss2     

# ''' loss2 v2 KL_abs loss'''
def KLDivAbsLoss(pred0, pred1, origin_classes):

    target = torch.pow(torch.exp(pred0), 1)            
    soft_target = target[:,0:origin_classes]/ torch.sum(target[:,0:origin_classes], dim = 1)[...,None]

    predict = torch.pow(torch.exp(pred1), 1)
    soft_predict = (predict/ torch.sum(predict, dim = 1)[...,None])[:,0:origin_classes]
    soft_predict = torch.log(soft_predict)

    kl_loss = nn.KLDivLoss(reduction = 'none')
    
    loss2 = kl_loss(soft_predict,soft_target)     
    loss2 = torch.abs(loss2)
    loss2 = torch.mean(torch.sum(loss2,dim = 1))
    
    return loss2

# '''loss2 v3 KD loss'''
def KDLoss(pred0, pred1, origin_classes):
    soft_target = nn.functional.softmax(pred0[:,0:origin_classes]  / 2, dim=1)
    # target = torch.pow(torch.exp(pred0), 0.5)            
    # soft_target = target[:,0:origin_classes]/ torch.sum(target[:,0:origin_classes], dim = 1)[...,None]

    soft_predict = nn.functional.softmax(pred1/ 2, dim=1)[:,0:origin_classes] 
    # predict = torch.pow(torch.exp(pred1), 0.5)
    # soft_predict = (predict/ torch.sum(predict, dim = 1)[...,None])[:,0:origin_classes]
    soft_predict = torch.log(soft_predict)

    loss2 = torch.mean( -1 * torch.sum(soft_target * soft_predict,dim = 1), dim = 0)  
    
    return loss2

def GeneralKDLoss(pred0, pred1, init_digit, end_digit):
    soft_target = nn.functional.softmax(pred0[:,init_digit:end_digit+1]  / 2, dim=1)
    # target = torch.pow(torch.exp(pred0), 0.5)            
    # soft_target = target[:,0:origin_classes]/ torch.sum(target[:,0:origin_classes], dim = 1)[...,None]

    soft_predict = nn.functional.softmax(pred1/ 2, dim=1)[:,init_digit:end_digit+1] 
    # predict = torch.pow(torch.exp(pred1), 0.5)
    # soft_predict = (predict/ torch.sum(predict, dim = 1)[...,None])[:,0:origin_classes]
    soft_predict = torch.log(soft_predict)

    loss2 = torch.mean( -1 * torch.sum(soft_target * soft_predict,dim = 1), dim = 0)  
    
    return loss2    
    
def CrossKDLoss(pred0, pred1, base_classes, incre_classes, prev_classes):
    
    s_digit = 0
    e_digit = base_classes-1
    T =2
    prev_task = int((prev_classes - base_classes) / incre_classes) + 1
    
    for t in range(1, prev_task+1):
        
        soft_target = nn.functional.softmax(pred0[:, s_digit : e_digit+1]  / T, dim=1)

        soft_predict = nn.functional.softmax(pred1/ T, dim=1)[:, s_digit : e_digit+1] 

        soft_predict = torch.log(soft_predict)

        kd_loss = torch.mean( -1 * torch.sum(soft_target * soft_predict,dim = 1), dim = 0)  
        
        if t==1:        
            loss2 = kd_loss            
        else:
            loss2+= kd_loss
        
        s_digit = e_digit + 1
        e_digit += incre_classes
    
    return loss2    

def KD_Sigmoid_Loss(pred0, pred1, origin_classes):
    
    sigmoid = nn.Sigmoid()
    T=2
    soft_target = sigmoid(pred0[:,0:origin_classes]  / T)
    soft_predict = sigmoid(pred1/ T)[:,0:origin_classes] 
    
    
    bce_loss = nn.BCELoss()


    loss = bce_loss(soft_predict, soft_target) 
    
    return loss

def combine_loss_with_ratio(loss1, loss2, ratio):
    
    return loss1 * (ratio) + loss2 * (1 - ratio)
    
def cvae_loss_function(recon_x, x, mu, logvar):
    # print(recon_x.shape,x.shape,mu.shape,logvar.shape)    
    mse = nn.MSELoss(reduction  = "sum")
    MSE = mse(x , recon_x)
    
    KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
    # print(BCE,KLD)
    
    return MSE , KLD        