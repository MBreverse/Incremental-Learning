from torchvision import datasets, models, transforms
from torchvision.models import vgg16, resnet50
import torch.nn as nn
import torch
import torch.nn.parameter as parameter


    
def init_weights_with_zeros(m):
    m.weight.data.fill_(0.)

def init_weights_with_ones(m):
    m.weight.data.fill_(1.)
    
class Resnet50_FeatureExtrator(nn.Module):

    def __init__(self):
        super().__init__()


        resnet = resnet50(pretrained = True)        

        resnet = nn.Sequential(*list(resnet.children())[:-1])
        
        self.model = resnet
        
        for name, value in self.model.named_parameters():
            value.requires_grad = False   
            
    def forward(self, x):

        feature = self.model(x)
        
        return feature
               
class CNN(nn.Module):
    def __init__(self, classes = 50):
        super().__init__()
        
        self.FE = Resnet50_FeatureExtrator()
        self.fc1 = nn.Linear(2048,1024)
        # self.fc2 = nn.Linear(1024,1024)
        self.fc2 = nn.Linear(1024,classes)
        
        
    def forward(self, x, mode = 'img'):
        
        if  mode == "img":
        
            feature = self.FE(x)
            feature = feature.detach().view(-1,2048)
            
        elif mode == "f":

            feature = x.view(-1,2048)
            
        resp = self.fc1(feature)
        # resp = nn.functional.dropout(resp, p=0.25, training = self.training, inplace=False) 
        resp = nn.functional.relu(resp, inplace=False) 
        output = self.fc2(resp)
        # resp = nn.functional.relu(resp, inplace=False)         
        # output = self.fc3(resp)
        
        return output

class ScaleKernel(nn.Module):
  def __init__(self, size_of_kernel):
    super().__init__()

    tensor = torch.rand(size_of_kernel)
    self.weight = parameter.Parameter(tensor, requires_grad=True)      

  def forward(self, x):

    y = x * self.weight 

    return y
    
class ShiftKernel(nn.Module):
  def __init__(self, size_of_kernel):
    super().__init__()

    tensor = torch.rand(size_of_kernel)
    self.weight = parameter.Parameter(tensor, requires_grad=True)      

  def forward(self, x):

    y = x + self.weight 

    return y
        
class Single_Scale_Shift_Kernel(nn.Module):
  def __init__(self, size_of_kernel, training = True):
    super().__init__()

    self.scale = ScaleKernel(size_of_kernel)
    self.shift = ShiftKernel(size_of_kernel) 
    
    self.scale.apply(init_weights_with_ones)
    self.shift.apply(init_weights_with_zeros)  
    
    for name, value in self.scale.named_parameters():
    
                if training == True:
                    value.requires_grad = True 
                else:
                    value.requires_grad = False    
    
    for name, value in self.shift.named_parameters():
    
                if training == True:
                    value.requires_grad = True 
                else:
                    value.requires_grad = False     
 
  def forward(self, x):

    y = self.scale(x)
    y = self.shift(y)

    return y
    
class Multi_Scale_Shift_Kernel(nn.Module):
  def __init__(self, insertion_check_list, sizes_of_kernels, training = True):
    super().__init__()
             
    self.kernel_list = nn.ModuleList([])  
    
    # print(insertion_check_list)
    sizes_of_kernels = iter(sizes_of_kernels)
    for insert_check in insertion_check_list:
    
        if insert_check == True:          
            size_of_kernel = next(sizes_of_kernels)
            # print(size_of_kernel)
            self.kernel_list.append( Single_Scale_Shift_Kernel(size_of_kernel))
            
        else:      
            
            self.kernel_list.append(None)
    # print(self.kernel_list)
             

  # indicate which kernel to be used
  def forward(self, x, index_of_kernel):
  
    if index_of_kernel <0:        
        raise "error: no kernel mapping of input index!!!"
        
    else:    
        selected_kernel = self.kernel_list[index_of_kernel]
        if selected_kernel!= None:
            #return shift and scale process output for kernel insertion here
            y = selected_kernel(x)
            
        else:
            #return input for no kernel insertion here
            y = x


    return y
     
class SS_Model(nn.Module):
    def __init__(self, basic_model, insertion_check_list, sizes_of_kernel_set, all_kernel_list = []):
        super().__init__()
        
        self.basic_model = basic_model
        self.num_fc_layer = len(list(self.basic_model.children())[1:])
        self.kernel_set1 = Multi_Scale_Shift_Kernel(insertion_check_list, sizes_of_kernel_set)  
        self.kernel_set2 = Multi_Scale_Shift_Kernel(insertion_check_list, sizes_of_kernel_set)
        self.all_kernel_set = nn.ModuleList(all_kernel_list)

    def forward(self, x, inp_mode = 'img'):
        
        if  inp_mode == "img":
        
            feature = self.basic_model.FE(x)
            feature = feature.detach().view(-1,2048) 
            
        elif inp_mode == "f":
            
            feature = x.view(-1,2048)
            
        output_list=[]
        # cls1 forward
        if self.kernel_set1 == None:
                                                 
            output1 = self.basic_model(feature, 'f')

        else:
            resp = feature 
            for i, module in enumerate(list(self.basic_model.children())[1:]):
                
                if i != (self.num_fc_layer-1):
                    #non-last layer forward
                    resp = self.kernel_set1(resp, 2*i)
                    resp = module(resp)
                    resp = self.kernel_set1(resp, 1 + 2*i)               
                    resp = nn.functional.relu(resp, inplace=False)
                else:
                    #last layer forward
                    resp = self.kernel_set1(resp, 2*i) 
                    output1 = module(resp)

         
        output_list.append(output1)
        
        if self.training ==True:
          
            resp = feature 
            for i, module in enumerate(list(self.basic_model.children())[1:]):
                
                if i != (self.num_fc_layer-1):
                    #non-last layer forward
                    resp = self.kernel_set2(resp, 2*i)
                    resp = module(resp)
                    resp = self.kernel_set2(resp, 1 + 2*i)               
                    resp = nn.functional.dropout(resp, 0.25, inplace=False)
                    resp = nn.functional.relu(resp, inplace=False)
                else:
                    #last layer forward
                    resp = self.kernel_set2(resp, 2*i) 
                    output2 = module(resp)
                    
            output_list.append(output2)
        else:
        
            # cls2 ~task N forward 
            for j in range(0,len(self.all_kernel_set)):
                                           
                ss_kernels = self.all_kernel_set[j]
                resp = feature 
     
                for i, module in enumerate(list(self.basic_model.children())[1:]):
                    
                    if i != (self.num_fc_layer-1):
                        #non-last layer forward
                        resp = ss_kernels(resp, 2*i)
                        resp = module(resp)
                        resp = ss_kernels(resp, 1 + 2*i)               
                        resp = nn.functional.relu(resp, inplace=False)
                    else:
                        #last layer forward
                        resp = ss_kernels(resp, 2*i) 
                        output = module(resp)
   
                output_list.append(output)
            
 
        return output_list

class AutoEncoder(nn.Module):

    def __init__(self):
        super().__init__()
            
        self.en1 = nn.Linear(2048,1024)
        self.en2 = nn.Linear(1024,512)
        self.en3 = nn.Linear(512,100)
                
        self.de1 = nn.Linear(100,512)
        self.de2 = nn.Linear(512,1024)
        self.de3 = nn.Linear(1024,2048)
        
        self.EncoderModules = nn.Sequential(self.en1, self.en2, self.en3)
        self.DecoderModules = nn.Sequential(self.de1, self.de2, self.de3)

    def forward(self, x):
        
        encode1 = self.en1(x)
        encode1 = torch.nn.functional.elu(encode1, alpha=1.0, inplace=False)
        
        encode2 = self.en2(encode1)    
        encode2 = torch.nn.functional.elu(encode2, alpha=1.0, inplace=False)
        
        encode3 = self.en3(encode2)  # last layer
        
        decode1 = self.de1(x)
        decode1 = torch.nn.functional.elu(decode1, alpha=1.0, inplace=False)
        
        decode2 = self.de2(decode1) 
        decode2 = torch.nn.functional.elu(decode2, alpha=1.0, inplace=False)
        
        decode3 = self.de3(decode2) # last layer
        
        return encode3, decode3
    
    def loss(self, pred, true, mode = 'mse'):
    
        if mode == "mse":
            mse_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            loss = mse_loss(pred, true)
        
        elif mode == "cross_entropy":
            en_loss = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')
            loss = en_loss(pred, true)
                        
        return loss
    
    def encode(self, x):
        '''Enocder used oly'''
        encode1 = self.en1(x)
        encode1 = torch.nn.functional.elu(encode1, alpha=1.0, inplace=False)
        
        encode2 = self.en2(encode1)    
        encode2 = torch.nn.functional.elu(encode2, alpha=1.0, inplace=False)
        
        encode3 = self.en3(encode2)  # last layer    
        
        return encode3

    def decode(self, x):
        '''Decoder used oly'''    
        decode1 = self.de1(x)
        decode1 = torch.nn.functional.elu(decode1, alpha=1.0, inplace=False)
        
        decode2 = self.de2(decode1) 
        decode2 = torch.nn.functional.elu(decode2, alpha=1.0, inplace=False)
        
        decode3 = self.de3(decode2) # last layer    

        return decode3                       