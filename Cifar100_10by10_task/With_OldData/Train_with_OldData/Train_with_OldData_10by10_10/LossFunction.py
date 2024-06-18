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
    
    kl_loss = nn.KLDivLoss(reduction = 'batchmean')
     
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