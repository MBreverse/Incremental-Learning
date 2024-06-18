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



class WeightScale(nn.Module):
  def __init__(self, size_of_kernel):
    super().__init__()

    tensor = torch.rand(size_of_kernel)
    self.weight = parameter.Parameter(tensor, requires_grad=True)      

  def forward(self, x):

    y = x * self.weight 

    return y
    
class WeightShift(nn.Module):
  def __init__(self, size_of_kernel):
    super().__init__()

    tensor = torch.rand([1, size_of_kernel]).T
    self.weight = parameter.Parameter(tensor, requires_grad=True)      

  def forward(self, x):

    y = torch.matmul(x, self.weight)
    # y = x * self.weight

    return y
 
class Single_Scale_Shift(nn.Module):
  def __init__(self, size_of_kernel, training = True):
    super().__init__()
    
    assert isinstance(size_of_kernel, tuple)
    
    self.scale = WeightScale(size_of_kernel[1])
    self.shift = WeightShift(size_of_kernel[0]) 
    
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
 
  def forward(self, x1, x2):

    y1 = self.scale(x1)
    y2 = self.shift(x2)

    return y1, y2
    
class Multi_Scale_Shift(nn.Module):
  def __init__(self, insertion_check_list, sizes_of_kernels, training = True):
    super().__init__()

    self.kernel_list = nn.ModuleList([])  
    
    # print(insertion_check_list)
    sizes_of_kernels = iter(sizes_of_kernels)
    for insert_check in insertion_check_list:
    
        if insert_check == True:          
            size_of_kernel = next(sizes_of_kernels)
            self.kernel_list.append( Single_Scale_Shift(size_of_kernel))       
        else:      
            self.kernel_list.append(None)

  # indicate which kernel to be used
  def forward(self, x1,x2, index_of_kernel):
  
    if index_of_kernel <0:        
        raise "error: no kernel mapping of input index!!!"
        
    else:    
        selected_kernel = self.kernel_list[index_of_kernel]
        if selected_kernel!= None:
            #return shift and scale process output for kernel insertion here
            y1, y2 = selected_kernel(x1, x2)
            
        else:
            #return input for no kernel insertion here
            y1 = x1 
            y2 = x2 

    return y1, y2 

class SS_Arch_2():
    def __init__(self):
    
        self.name = "SS forward method"

    def FcLayerDecomposition(self, FCModel):        
  
        if isinstance(FCModel, list):
            fc_module_list = FCModel
        else:
            fc_module_list = list(FCModel.children())

        num_layers = len(fc_module_list)
        # activation_list = 
        if fc_module_list == [] and num_layers == 0 :
            
            fc_module_list = [FCModel] 
            num_layers = 1 
        
        Decomp_FC_dict = {"fc_list" :fc_module_list,
                          # "activ" : activation_list,
                          "fc_num" : num_layers}
        
        return Decomp_FC_dict

    def Single_SS_forward(self, input, module, ss_set, idx):
        
        resp1, resp2 = ss_set(resp, 2*idx) # resp1[1,n], resp2[1,1]
        resp = module(resp1) # resp[1,m]                
        resp = resp + resp2  # resp[1,m]
        
        return resp

    def SSFC_forward(self, input, FCModel, ss_set):

        FC_dict = self.FcLayerDecomposition(FCModel)
        
        resp = input

        for i, module in enumerate(FC_dict["fc_list"]):
                  
            if i != (FC_dict["fc_num"] - 1):            
                #non-last layer forward
                
                resp1 = module(resp) # resp[1,m]  
                resp2 = resp         # resp[1,n]
                resp1, resp2 = ss_set(resp1, resp2, i) # resp1[1,m], resp2[1,1]       
                resp = resp1 + resp2  # resp[1,m]

                resp = nn.functional.dropout(resp, p=0.25, training = module.training, inplace=False)                
                resp = nn.functional.relu(resp, inplace=False) # need post-process while assign base model
            
            else:            
                #last layer forward
                resp1 = module(resp) # resp[1,m]  
                # resp2 = resp         # resp[1,n]
                # resp1, resp2 = ss_set(resp1, resp2, i) # resp1[1,m], resp2[1,1]                  
                output = resp1  # resp[1,m]
    
        return output

    def Base_forward(self, input, Base):
    
        FC_dict = self.FcLayerDecomposition(Base)
        
        resp = input

        for i, module in enumerate(FC_dict["fc_list"]):
                  
            if i != (FC_dict["fc_num"] - 1):            
                #non-last layer forward
                
                resp = module(resp) # resp[1,m]  
                resp = nn.functional.dropout(resp, p=0.25, training = module.training, inplace=False)                
                resp = nn.functional.relu(resp, inplace=False) # need post-process while assign base model
            
            else:            
                #last layer forward
                output = module(resp)

    
        return output        
        
    def FullSS_forward(self, input, Base, all_ss_set):

        output_list = []
        output_list.append(self.Base_forward(input, Base))        
        
        # iterate ss correspond to task
        for j in range(0, len(all_ss_set)):
                                       
            ss_set = all_ss_set[j]

            output = self.SSFC_forward(input, Base, ss_set)

            output_list.append(output)
        # print(len(output_list))
        return output_list
    
    def FullTask_SS_forward(self, input, Base, all_ss_set):

        output_list = []
        output_list.append(self.Base_forward(input, Base))        
        
        # iterate ss correspond to task
        for j in range(0, len(all_ss_set)):
                                       
            ss_set = all_ss_set[j]

            output = self.SSFC_forward(input, Base, ss_set)

            output_list.append(output)
        # print(len(output_list))
        return output_list

    def forward(self, input, Base, ss_set1, ss_set2):  

        if ss_set1 == None:
            out1 = self.Base_forward(input, Base)
        else:
            out1 = self.SSFC_forward(input, Base, ss_set1)
            
        out2 = self.SSFC_forward(input, Base, ss_set2)
        
        return out1, out2

class SS_Model_2(nn.Module):
    def __init__(self, basic_model, insertion_check_list, sizes_of_kernel_set, all_kernel_list = []):
        super().__init__()
        
        self.basic_model = basic_model
        self.num_fc_layer = len(list(self.basic_model.children())[1:])
        self.kernel_set1 = Multi_Scale_Shift(insertion_check_list, sizes_of_kernel_set)  
        self.kernel_set2 = Multi_Scale_Shift(insertion_check_list, sizes_of_kernel_set)
        self.all_kernel_set = nn.ModuleList(all_kernel_list)
        self.arch2 = SS_Arch_2()

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
            output1 = self.arch2.SSFC_forward(feature, list(self.basic_model.children())[1:], self.kernel_set1)
         
        output_list.append(output1)
        
        if self.training ==True:
        
            output2 = self.arch2.SSFC_forward(feature, list(self.basic_model.children())[1:], self.kernel_set2)
            output_list.append(output2)
        else:
            output_list = self.arch2.FullTask_SS_forward(feature, list(self.basic_model.children())[1:], self.all_kernel_set)

        return output_list



class Recoverable_Linear(nn.Linear):
  def __init__(self, in_features, out_features, bias=True, device=None, dtype=None):
    super().__init__(in_features, out_features, bias, device, dtype)
    
    self.prev_weight = self.weight.detach().clone()
    self.prev_bias = self.bias.detach().clone()  

  def recover(self, dim = 1, interval = []):

    assert len(interval) == 2

    start, end = interval
    cur_state_dict = self.state_dict()
    cur_state_dict["weight"][start:end] = self.prev_weight[start:end]
    cur_state_dict["bias"][start:end] = self.prev_bias[start:end]
    self.load_state_dict(cur_state_dict)
  
  def update_init(self):
    self.prev_weight = self.weight.detach().clone()
    self.prev_bias = self.bias.detach().clone()    

class Recoverable_CNN(nn.Module):
    def __init__(self, classes = 50):
        super().__init__()
        
        self.FE = Resnet50_FeatureExtrator()
        self.fc1 = Recoverable_Linear(2048,1024)
        self.fc2 = Recoverable_Linear(1024,classes)
  
    def forward(self, x, mode = 'f'):
        
        if  mode == "img":
            feature = self.FE(x)
            feature = feature.detach().view(-1,2048)
            
        elif mode == "f":

            feature = x.view(-1,2048)
            
        resp = self.fc1(feature)
        resp = nn.functional.dropout(resp, p=0.25, training = self.training, inplace=False) 
        resp = nn.functional.relu(resp, inplace=False) 
        output = self.fc2(resp)

        return output    

class Scale_Shift_Set(nn.Module):
    def __init__(self, m_size, n_size):
        super().__init__() 
        # self.alpha1 = parameter.Parameter(torch.rand([1, n_size]), requires_grad=True)
        # self.alpha2 = parameter.Parameter(torch.rand([1, m_size]), requires_grad=True)
        # self.beta1 = parameter.Parameter(torch.rand([1, n_size]).T, requires_grad=True)
        # self.beta2 = parameter.Parameter(torch.rand([1, m_size]), requires_grad=True)
        self.alpha1 = parameter.Parameter(torch.rand([1, n_size]).fill_(1.), requires_grad=True)
        self.alpha2 = parameter.Parameter(torch.rand([1, m_size]).fill_(1.), requires_grad=True)
        self.beta1 = parameter.Parameter(torch.rand([1, n_size]).fill_(0.).T, requires_grad=True)
        self.beta2 = parameter.Parameter(torch.rand([1, m_size]).fill_(0.), requires_grad=True)

class SS_Arch_Forward():
    
    def forward(self, x, SS, linear_m):
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        weight = linear_m.get_parameter('weight')
        bias = linear_m.get_parameter('bias')    
        m_size = linear_m.weight.data.size(0) # oup size
        n_size = linear_m.weight.data.size(1) # inp size        
        
        if SS != None:

            alpha1 = SS.get_parameter('alpha1')    
            alpha2 = SS.get_parameter('alpha2')
            beta1 = SS.get_parameter('beta1')
            beta2 = SS.get_parameter('beta2')    
            
            resp1 = torch.matmul( weight, (x * alpha1).T).T * alpha2
            resp2 = torch.matmul(x, beta1)
            resp3 = torch.matmul(x, torch.ones([n_size,1]).to(device))
            resp3 = torch.matmul(resp3, beta2)
            output = resp1  + resp2 + resp3 +  bias
        else:
            output = torch.matmul( weight, x.T).T
            
        return output

class Multi_SS_Manager(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.multi_SS = nn.ModuleList([])
    
    def build_ss(self, module, flag = True):
    
        assert isinstance(module, nn.Linear)
        m_size = module.weight.data.size(0) # oup size
        n_size = module.weight.data.size(1) # inp size
        
        try:
            assert flag
            SS = Scale_Shift_Set(m_size, n_size)        
            self.multi_SS.append(SS)                       
        except:
            self.multi_SS.append(None)
            
        return self.multi_SS[-1]
        
    def load_ss(self, ss_state_dict):
        
        self.multi_SS.load_state_dict(ss_state_dict)
        
class SS_Model3(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        
        self.base_model = base_model
        self.ss_manager = Multi_SS_Manager()
        self.build_SS()
        self.arch = SS_Arch_Forward()
        
    def forward(self, x, mode ='f'):
        
        assert self.ss_manager != None
                
        if mode == 'x':
            resp = self.base_model.FE(x).detach().view(-1,2048)
       
        if mode == 'f':
            resp = x.view(-1,2048)
        
        
        for i, key in enumerate(self.module_dict.keys()):
            
            ss, module = self.module_dict[key]
            
            resp = self.arch.forward(resp, ss, module)    
            
            if i < len(self.module_dict)-1:       
            
                resp = nn.functional.dropout(resp, p = 0.25, training = self.training, inplace=False) 
                resp = nn.functional.relu(resp, inplace=False)
                
        return resp

    def build_SS(self):        
        
        self.module_dict = {}
        count = 0
        for idx, module in enumerate(self.base_model.children()):           
            
            if isinstance(module, nn.Linear):            
                ss = self.ss_manager.build_ss(module, count!=1)
                self.module_dict.update({str(idx):[ss, module]})                
                count += 1
                
    def get_SS_dict(self):
        
        return self.ss_manager.get_submodule("multi_SS").state_dict()
    
    def load_SS(self, state_dict):

        self.ss_manager.load_ss(state_dict)


class CVAE(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        
        self.feature_size = feature_size
        self.class_size = class_size

        # encode        
        self.encoder = CVAEEncoder(latent_size, feature_size, class_size)

        # decode
        self.decoder = CVAEDecoder(latent_size, feature_size, class_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        z_mu, z_var = self.encoder(x, c)
        
        return z_mu, z_var

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        return mu + eps*std

    def decode(self, z, c): # P(x|z, c)
        '''
        z: (bs, latent_size)
        c: (bs, class_size)
        '''
        xhead = self.decoder(z, c)
        sigmoid = torch.nn.Sigmoid()
        xhead = sigmoid(xhead)
        
        # no sigmoid because of Cifar100
        
        return xhead#self.sigmoid()

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.feature_size), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar        

class CVAEEncoder(nn.Module):
    def __init__(self, latent_size, feature_size, class_size):
        super().__init__()  
        
        hidden_dim = 100
        self.ShareFC  = nn.Linear(feature_size + class_size, hidden_dim)
        self.MeanEncoder = nn.Linear(hidden_dim, latent_size)
        self.VarEncoder = nn.Linear(hidden_dim, latent_size)

    
    def forward(self, x, c):
        '''Decoder used oly'''  
        
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)        
        h1 = self.ShareFC(inputs)
        h1 = torch.nn.functional.elu(h1, alpha=1.0, inplace=False) 
        z_mu = self.MeanEncoder(h1)
        z_var = self.VarEncoder(h1) 
        
        return  z_mu, z_var

class CVAEDecoder(nn.Module):
    def __init__(self, laten_size, feature_size, class_size):
        super().__init__()  
        
        hidden_dim = 100
        self.de1 = nn.Linear(laten_size + class_size, hidden_dim)
        self.de2 = nn.Linear(hidden_dim,feature_size)
    
    def forward(self, z, c):
        '''Decoder used oly'''  
        
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        
        decode1 = self.de1(inputs)
        decode1 = torch.nn.functional.elu(decode1, alpha=1.0, inplace=False)        
        decode2 = self.de2(decode1) # last layer    
        
        return decode2    
        