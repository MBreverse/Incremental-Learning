import torch
import torch.nn as nn
import torch.nn.parameter as parameter
from ModelCollection import Resnet50_FeatureExtrator
# from SupportFunction import *
# from DatasetProcess import *

# cnn = Recoverable_CNN()
# ssm = SS_Model3(cnn)
# ssm.build_SS()

# print(ssm.ss_manager.state_dict().keys())
# print(ssm.get_submodule("ss_manager").get_submodule("multi_SS"))


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
                ss = self.ss_manager.build_ss(module, count!=-1)
                self.module_dict.update({str(idx):[ss, module]})                
                count += 1
                
    def get_SS_dict(self):
        
        return self.ss_manager.get_submodule("multi_SS").state_dict()
    
    def load_SS(self, state_dict):

        self.ss_manager.load_ss(state_dict)


def test():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    cnn = Recoverable_CNN(50).to(device)
    model= SS_Model3(cnn).to(device)
    model(torch.rand([1, 2048]).to(device), "f")
    # print(model.ss_manager)
    # torch.save( model.get_SS_dict(), "./ss.pth.tar")
    # ss_dict = torch.load( "./ss.pth.tar")
    # model.load_SS(ss_dict)
    


# test()


import torch.optim as optim

a = torch.rand([1])
opt = optim.Adam([a], lr = 0.01)
print(opt.state_dict()['param_groups'])

origin_dict = opt.state_dict()
for param in origin_dict['param_groups']:    
    param.update({"lr":0.001})
opt.load_state_dict(origin_dict)
