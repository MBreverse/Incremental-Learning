from torchvision import datasets, models, transforms
from torchvision.models import vgg16, resnet50
import torch.nn as nn
import torch
import torch.nn.parameter as parameter

    
class FeatureExtractor(nn.Module):
   
    def __init__(self, mode):
        super().__init__()
        
        if mode == "res50":
            self.model = Resnet50_FeatureExtrator()
        
        if mode == "vgg16":
            self.model = None # update after
    def forward(self, x):
        return sself.model(x)
       
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
        resp = nn.functional.dropout(resp, p=0.25, training = self.training, inplace=False)              
        output = self.fc2(resp)
        # resp = nn.functional.relu(resp, inplace=False)         
        # output = self.fc3(resp)
        
        return output

class AutoEncoderWithCLF(nn.Module):

    def __init__(self, base_class = 50):
        super().__init__()
            
        self.en1 = nn.Linear(2048,1024)
        self.en2 = nn.Linear(1024,512)
        self.en3 = nn.Linear(512,base_class)
                
        self.de1 = nn.Linear(base_class,512)
        self.de2 = nn.Linear(512,1024)
        self.de3 = nn.Linear(1024,2048)
        
        # self.EncoderModules = nn.Sequential(self.en1, self.en2, self.en3)
        # self.DecoderModules = nn.Sequential(self.de1, self.de2, self.de3)
        self.clf = nn.Linear(100, 50)

    def forward(self, x):

        z = self.encode(x)
        x_head = self.decode(z)                   
        cls_pred = self.clf(z)
        
        return z, x_head, cls_pred
    
    def loss(self, pred, true, origin_classes = 0, mode = 'mse'):
    
        if mode == "mse":
            mse_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            loss = mse_loss(pred, true)
        
        elif mode == "cross_entropy":
            en_loss = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')
            loss = en_loss(pred, true)
            
        elif mode == "bce":
            # bce_loss = torch.nn.BCELoss()
            bce_loss = torch.nn.BCEWithLogitsLoss()            
            loss = bce_loss(pred, true.float())
            
        elif mode == "bce_kd":
            bce_loss = torch.nn.BCEWithLogitsLoss()
            sigmoid_module = nn.Sigmoid()
            true = sigmoid_module(true)
            loss = bce_loss(pred[:,0:origin_classes], true)
                        
        return loss
    
    def encode(self, x):
        '''Enocder used oly'''
        encode1 = self.en1(x)
        encode1 = nn.functional.dropout(encode1, p=0.25, training = self.training, inplace=False)  
        encode1 = torch.nn.functional.elu(encode1, alpha=1.0, inplace=False)
        
        encode2 = self.en2(encode1)   
        encode2 = nn.functional.dropout(encode2, p=0.25, training = self.training, inplace=False)          
        encode2 = torch.nn.functional.elu(encode2, alpha=1.0, inplace=False)
        
        encode3 = self.en3(encode2)  # last layer    

        return encode3

    def decode(self, x):
        '''Decoder used oly'''  
        # x = torch.nn.functional.sigmoid(x)
        # x = torch.nn.functional.elu(x, alpha=1.0, inplace=False)
        decode1 = self.de1(x)
        # decode1 = nn.functional.dropout(decode1, p=0.25, training = self.training, inplace=False)  
        decode1 = torch.nn.functional.elu(decode1, alpha=1.0, inplace=False)
        
        decode2 = self.de2(decode1) 
        # decode2 = nn.functional.dropout(decode2, p=0.25, training = self.training, inplace=False)  
        decode2 = torch.nn.functional.elu(decode2, alpha=1.0, inplace=False)
        
        decode3 = self.de3(decode2) # last layer    

        return decode3   
    
    def classify(self, z):
    
        z = torch.nn.functional.elu(z, alpha=1.0, inplace=False)
        cls_pred = self.clf(z)
        
        
        return cls_pred

class AutoEncoderwithFE(nn.Module):
    
    def __init__(self, base_class = 50, feature_size = 2048):
        super().__init__()
        
        self.FE = Resnet50_FeatureExtrator()
        self.AE = AutoEncoder(base_class)
        self.feature_size = feature_size
        
    def forward(self, x, inp_mode = 'img'):
        
        if inp_mode == 'img':
            
            feature = self.FE(x)
            feature = feature.detach().view(-1,self.feature_size )
        
        elif inp_mode == 'f':
            feature = x.view(-1,self.feature_size )
        
        
        encode, decode = self.AE(feature)
        
        return encode, decode
        
    def encode(self, x,  inp_mode = 'img'):
    
        if inp_mode == 'img':
            
            feature = self.FE(x)
            feature = feature.detach().view(-1,self.feature_size )
        
        elif inp_mode == 'f':
            feature = x.view(-1,self.feature_size )    
        
        zcode = self.AE.encode(feature)
        
        return zcode
        
    def decode(self, x):   
        
        return self.AE.decode(x)      

class Encoder(nn.Module):
    def __init__(self, base_class):
        super().__init__()  
        
        self.en1 = nn.Linear(2048,1024)
        self.en2 = nn.Linear(1024,512)
        self.en3 = nn.Linear(512,base_class)
    
    def forward(self, x):
        encode1 = self.en1(x)
        encode1 = nn.functional.dropout(encode1, p=0.25, training = self.training, inplace=False)  
        encode1 = torch.nn.functional.elu(encode1, alpha=1.0, inplace=False)
        
        encode2 = self.en2(encode1)   
        encode2 = nn.functional.dropout(encode2, p=0.25, training = self.training, inplace=False)          
        encode2 = torch.nn.functional.elu(encode2, alpha=1.0, inplace=False)        
        encode3 = self.en3(encode2)
        
        return encode3

class Decoder(nn.Module):
    def __init__(self, base_class):
        super().__init__()  
        
        self.de1 = nn.Linear(base_class,512)
        self.de2 = nn.Linear(512,1024)
        self.de3 = nn.Linear(1024,2048)
    
    def forward(self, x):
        '''Decoder used oly'''  

        decode1 = self.de1(x)
        decode1 = torch.nn.functional.elu(decode1, alpha=1.0, inplace=False)
        
        decode2 = self.de2(decode1) 
        decode2 = torch.nn.functional.elu(decode2, alpha=1.0, inplace=False)
        
        decode3 = self.de3(decode2) # last layer    
        
        return decode3    

class AutoEncoder(nn.Module):

    def __init__(self, base_class = 50):
        super().__init__()
                
        self.Encoder = Encoder(base_class)
        self.Decoder = Decoder(base_class)
        
    def forward(self, x):

        z = self.encode(x)
        x_head = self.decode(z)                   
        
        return z, x_head

    def encode(self, x): 
        
        z = self.Encoder(x)   

        return z

    def decode(self, x):
        
        x_h = self.Decoder(x)

        return x_h   
        
    def loss(self, pred, true, origin_classes = 0, mode = 'mse'):
    
        if mode == "mse":
            mse_loss = torch.nn.MSELoss(size_average=None, reduce=None, reduction='mean')
            loss = mse_loss(pred, true)
        
        elif mode == "cross_entropy":
            en_loss = torch.nn.CrossEntropyLoss(weight=None, reduction='mean')
            loss = en_loss(pred, true)
            
        elif mode == "bce":
            # bce_loss = torch.nn.BCELoss()
            bce_loss = torch.nn.BCEWithLogitsLoss()            
            loss = bce_loss(pred, true.float())
            
        elif mode == "bce_kd":
            bce_loss = torch.nn.BCEWithLogitsLoss()
            sigmoid_module = nn.Sigmoid()
            true = sigmoid_module(true)
    
            loss = bce_loss(pred[:,0:origin_classes], true[:,0:origin_classes])
                        
        return loss
    

        '''Decoder used oly'''    
        decode1 = self.de1(x)
        decode1 = torch.nn.functional.elu(decode1, alpha=1.0, inplace=False)
        
        decode2 = self.de2(decode1) 
        decode2 = torch.nn.functional.elu(decode2, alpha=1.0, inplace=False)
        
        decode3 = self.de3(decode2) # last layer    

        return decode3                       
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


if __name__ == "__main__" :
   
   print(__name__)