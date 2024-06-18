from torchvision import datasets, models, transforms
from torchvision.models import vgg16, resnet50
import torch.nn as nn
import torch
import torch.nn.parameter as parameter

device = "cuda" if torch.cuda.is_available() else "cpu"

    
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

class VAE(nn.Module):
    
    def __init__(self):
        super().__init__()
        
        self.SharedEncoder = nn.Linear(784,400)
        self.MeanEncoder = nn.Linear(400,20)
        self.VarEncoder = nn.Linear(400,20)
        
        self.Decoder_Layer1 = nn.Linear(20,400)
        self.Decoder_Layer2 = nn.Linear(400,784)
    
    def reparameter(self, mean, logvar):
        
        std = torch.exp(0.5 * logvar)
        
        eps = torch.randn_like(std)
        # torch.normal(mean, std, *, generator=None, out=None) 
        
        return mean + std * eps
    
    def decode(self, z):
        
        resp = self.Decoder_Layer1(z)
        resp = torch.nn.functional.relu(resp)
        recover_x = self.Decoder_Layer2(resp)
        
        recover_x = torch.nn.functional.sigmoid(recover_x)
        
        return recover_x
    
    def generate(self, batch_size):
        
        z = torch.randn([batch_size, 20]).to("cuda")
        x = self.decode(z)
        return x
        
    
    def forward(self, x):
        
        x = x.view(-1, 784)
        resp =  self.SharedEncoder(x)
        resp = torch.nn.functional.relu(resp)
        
        en_mean = self.MeanEncoder(resp)
        en_var = self.VarEncoder(resp)
        
        z = self.reparameter(en_mean, en_var)
        # z_max, _ = torch.max(z,1)
        # z_max = z_max.unsqueeze(dim=1)
        
        # z_min, _ = torch.min(z, 1)
        # z_min = z_min.unsqueeze(dim=1)
        # z = (z - z_min) / (z_min)
        
        # resp = self.Decoder_Layer1(z)
        # resp = torch.nn.functional.leaky_relu(resp)
        # recover_x = self.Decoder_Layer2(resp)
        recover_x = self.decode(z)
        
        return en_mean, en_var, z, recover_x

class CVAE_x(nn.Module):
    def __init__(self, feature_size, latent_size, class_size):
        super(CVAE, self).__init__()
        self.feature_size = feature_size
        self.class_size = class_size

        # encode
        self.fc1  = nn.Linear(feature_size + class_size, 400)
        self.fc21 = nn.Linear(400, latent_size)
        self.fc22 = nn.Linear(400, latent_size)

        # decode
        self.fc3 = nn.Linear(latent_size + class_size, 400)
        self.fc4 = nn.Linear(400, feature_size)

        self.elu = nn.ELU()
        self.sigmoid = nn.Sigmoid()

    def encode(self, x, c): # Q(z|x, c)
        '''
        x: (bs, feature_size)
        c: (bs, class_size)
        '''
        inputs = torch.cat([x, c], 1) # (bs, feature_size+class_size)
        h1 = self.elu(self.fc1(inputs))
        z_mu = self.fc21(h1)
        z_var = self.fc22(h1)
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
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        h3 = self.elu(self.fc3(inputs))
        return self.sigmoid(self.fc4(h3))

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 28*28), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar        

class SS_Arch_1():
    def __init__(self):
    
        self.name = "SS forward method"
        
    def FcLayerDecomposition(self, FCModel):        
  
        
        fc_module_list = list(FCModel.children())
        # print(FCModel)
            
        num_layers = len(fc_module_list)
        # activation_list = 
        if fc_module_list == [] and num_layers == 0 :
            
            fc_module_list = [FCModel] 
            num_layers = 1 
        
        Decomp_FC_dict = {"fc_list" :fc_module_list,
                          # "activ" : activation_list,
                          "fc_num" : num_layers}
        
        return Decomp_FC_dict

    def SSFC_forward(self, input, FCModel, ss_set):

        FC_dict = self.FcLayerDecomposition(FCModel)
        
        resp = input

        
        for i, module in enumerate(FC_dict["fc_list"]):
                  
            if i != (FC_dict["fc_num"] - 1):            
                #non-last layer forward
              
                resp = ss_set(resp, 2*i)
                resp = module(resp)                
                resp = ss_set(resp, 1 + 2*i) 
                
                resp = nn.functional.dropout(resp, p=0.25, training = FCModel.training, inplace=False)                
                resp = nn.functional.elu(resp, inplace=False) # need post-process while assign base model
            
            else:            
                #last layer forward
                resp = ss_set(resp, 2*i) 
                output = module(resp)    
    
        return output

    def FullSS_forward(self, input, Base, all_ss_set):

        output_list = []
        output_list.append(Base(input))        
        
        # iterate ss correspond to task
        for j in range(0, len(all_ss_set)):
                                       
            ss_set = all_ss_set[j]

            output = self.SSFC_forward(input, Base, ss_set)

            output_list.append(output)
        # print(len(output_list))
        return output_list

    def forward(self, input, Base, ss_set1, ss_set2):  

        if ss_set1 == None:
            out1 = Base(input)
        else:
            out1 = self.SSFC_forward(input, Base, ss_set1)
            
        out2 = self.SSFC_forward(input, Base, ss_set2)
        
        return out1, out2

class SSAEwithFE(nn.Module):
    def __init__(self, AEwithFE, insertion_check_list, sizes_of_ss_set, all_ss_list = []):
        super().__init__()
        
        self.Base = AEwithFE
        self.BaseFE = self.Base.FE
        self.BaseEncoder = self.Base.AE.Encoder
        self.BaseDecoder = self.Base.AE.Decoder
        self.feature_size = 2048
        
        self.en_ss_set1 = Multi_Scale_Shift_Kernel(insertion_check_list, sizes_of_ss_set)  
        self.en_ss_set2 = Multi_Scale_Shift_Kernel(insertion_check_list, sizes_of_ss_set)
        self.de_ss_set1 = Multi_Scale_Shift_Kernel(insertion_check_list, sizes_of_ss_set)  
        self.de_ss_set2 = Multi_Scale_Shift_Kernel(insertion_check_list, sizes_of_ss_set)      
        
        self.all_ss_set = nn.ModuleList(all_ss_list)         
                
        self.arch = SS_Arch_1()
    
    def single_encode(self, x, mode = 'img'):
        
        if mode == 'img':            
            feature = self.BaseFE(x)
            feature = feature.detach().view(-1,self.feature_size )
        
        elif mode == 'f':
            feature = x.view(-1,self.feature_size )    

        zcode = self.arch.SSFC_forward(feature, self.BaseEncoder, self.en_ss_set2 )     

        return zcode 
       
    def encode(self, x, mode = 'img'):
    
        if mode == 'img':
            
            feature = self.BaseFE(x)
            feature = feature.detach().view(-1,self.feature_size )
        
        elif mode == 'f':
            feature = x.view(-1,self.feature_size )    

        zcodes = self.arch.forward(feature, self.BaseEncoder, self.en_ss_set1, self.en_ss_set2)     

        return zcodes
        
    def decode(self, x):   
    
        xhead = self.arch.SSFC_forward(x, self.BaseDecoder, self.de_ss_set2)
                
        return xhead        
        
    def forward(self, x, mode = 'img', out_mode = 'both'):
        
        if mode == 'img':
            
            feature = self.BaseFE(x)
            feature = feature.detach().view(-1,self.feature_size)
        
        elif mode == 'f':
            feature = x.view(-1,self.feature_size)
        
        if self.training:
        
            encode_vectors = self.encode(x)
            decode_vector = self.decode(encode_vectors[-1])
            
            return encode_vectors, decode_vector 
            
        else:
        
            if out_mode == 'both':
                
                # hireachy testing
                encode_vectors = self.arch.FullSS_forward(feature, self.BaseEncoder, self.all_ss_set)
                decode_vector = self.decode(encode_vectors[-1])        
            
                return encode_vectors, decode_vector    

            elif out_mode == 'encode':
               
                # hireachy testing
                encode_vectors = self.arch.FullSS_forward(feature, self.BaseEncoder, self.all_ss_set)        
                return encode_vectors
           
            elif out_mode == 'decode':
                decode_vector = self.decode(encode_vectors[-1])        
                return decode_vector               

class Block_MultiSS(nn.Module):
    
    def __init__(self, model_block_names, inser_check_dict, ss_size_dict):
        super(Block_MultiSS, self).__init__()
        
        self.SSDict = nn.ModuleDict()
        
        for name in model_block_names:

            self.SSDict.update({name : Multi_Scale_Shift_Kernel(inser_check_dict[name], ss_size_dict[name])})
    
    def forward(self, name, ss_index, x):
        
        return self.SSDict[name](ss_index, x)

class SSCVAE(nn.Module):
    def __init__(self, cvae, insert_check_dict, sizes_dict):
        super().__init__()
        
        self.Base = cvae
        # self.BaseFE = self.Base.FE
        self.BaseEncoder = self.Base.encoder
        self.BaseDecoder = self.Base.decoder
        
        self.feature_size = 784
        
        encodr_block_names = ["share", "mean", "var"]
        self.EncoderSS1 = Block_MultiSS(encodr_block_names, insert_check_dict["en"], sizes_dict["en"])  
        # self.EncoderSS2 = Block_MultiSS(encodr_block_names, insert_check_dict["en"], sizes_dict["en"])  

        decodr_block_names = ["de"]
        self.DecoderSS1 = Block_MultiSS(decodr_block_names, insert_check_dict, sizes_dict)  
        # self.DecoderSS2 = Block_MultiSS(decodr_block_names, insert_check_dict["de"], sizes_dict["de"])  
        
        # self.All_SS_Set = nn.ModuleList(all_ss_list)         
                
        self.arch = SS_Arch_1()    

    def encode(self, x, c):
        
        input = torch.cat([x, c], 1)
        
        resp = self.arch.SSFC_forward( input, self.BaseEncoder.ShareFC, self.EncoderSS1.SSDict["share"])                
        resp = nn.functional.dropout(resp, p=0.25, training = self.training, inplace=False)                
        resp = nn.functional.elu(resp, inplace=False) # need post-process while assign base model        
        
        z_mu = self.arch.SSFC_forward( resp, self.BaseEncoder.MeanEncoder, self.EncoderSS1.SSDict["mean"])
        z_var = self.arch.SSFC_forward( resp, self.BaseEncoder.MeanEncoder, self.EncoderSS1.SSDict["var"])
        
        return z_mu, z_var    
    
    def decode(self, x, c):
        
        input = torch.cat([x, c], 1)
        xhead = self.arch.SSFC_forward( input, self.BaseDecoder, self.DecoderSS1.SSDict["de"])
        
        return self.Base.sigmoid(xhead)
    
    def reparameterize(self, mu, logvar):
    
        std = torch.exp(0.5*logvar)
        eps = torch.randn_like(std)
        
        return mu + eps*std
        
    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, 28*28), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar   

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
        
        # no sigmoid because of Cifar100
        
        return xhead#self.sigmoid()

    def forward(self, x, c):
        mu, logvar = self.encode(x.view(-1, self.feature_size), c)
        z = self.reparameterize(mu, logvar)
        return self.decode(z, c), mu, logvar        

class CVAEEncoder(nn.Module):
    def __init__(self, latent_size, feature_size, class_size):
        super().__init__()  
        
        hidden_dim = 400
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
        
        hidden_dim = 400
        self.de1 = nn.Linear(laten_size + class_size, hidden_dim)
        self.de2 = nn.Linear(hidden_dim,feature_size)
    
    def forward(self, z, c):
        '''Decoder used oly'''  
        
        inputs = torch.cat([z, c], 1) # (bs, latent_size+class_size)
        
        decode1 = self.de1(inputs)
        decode1 = torch.nn.functional.elu(decode1, alpha=1.0, inplace=False)        
        decode2 = self.de2(decode1) # last layer    
        
        return decode2    


if __name__ == "__main__":
    print(__name__)