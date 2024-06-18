from torchvision import datasets, models, transforms
import torchvision
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch
from torch.distributions.multivariate_normal import MultivariateNormal

from PIL import Image
import numpy as np
import sys
import os
import random

from ModelCollection import Resnet50_FeatureExtrator, VAE
from SupportFunction import *
from Config import *

device = "cuda" if torch.cuda.is_available() else "cpu"


class Mnist_Split(datasets.MNIST):   

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False, digits = [0,1], real_label = True):
        super(Mnist_Split, self).__init__(root, train, transform, target_transform, download)
        
        
        #get only the two digits
        self.digit_labels = None
        self.digit_data = None
        self.classes= digits 
        
        if self.train:
            
            #loop over the given digits and extract there corresponding data
            for digit in digits:
            
                digit_mask = torch.eq(self.train_labels , digit) # 判斷train_label中的類別是否對應指定的digit，是=True, 不是=False
                digit_index = torch.nonzero(digit_mask)  # 取出digit_mask=true的label的索引值(指定labe對應的索引直)
                digit_index = digit_index.view(-1)    # 轉成一維tensor(digit_index原本就是一維陣列)
                this_digit_data = self.train_data[digit_index]    # 取出指定的data
                this_digit_labels = self.train_labels[digit_mask]    # 取出指定的label值(創造出足夠的sample空間)
                
                # training data要重新定義label
                if not real_label:
                    this_digit_labels.fill_(digits.index(digit))    # 重新定義label值(從0開始)
                if self.digit_data is None:
                    self.digit_data=this_digit_data.clone()    # 複製張量(梯度狀態一樣複製但數值無法流入)
                    self.digit_labels=this_digit_labels.clone()
                else:#concate data after first assign with first digit data
                    self.digit_data=torch.cat((self.digit_data,this_digit_data),0)
                    self.digit_labels=torch.cat((self.digit_labels,this_digit_labels),0)
               

            #self.train_data, self.train_labels = torch.load(
                #os.path.join(root, self.processed_folder, self.training_file))
        else:
                       #loop over the given digits and extract there corresponding data
            for digit in digits:
                digit_mask = torch.eq(self.test_labels , digit)    # 比較兩tensor是否相等
                digit_index = torch.nonzero(digit_mask)    # 非0元素的索引值(由左~右, 上~下讀取一個陣列)
                digit_index = digit_index.view(-1)
                this_digit_data = self.test_data[digit_index]
                this_digit_labels = self.test_labels[digit_mask]
                
                # 創建不為真值的label(label在每個task都重新從0開始標籤)
                if not real_label:
                    this_digit_labels.fill_(digits.index(digit))
                if self.digit_data is None:
                    self.digit_data = this_digit_data.clone()
                    self.digit_labels = this_digit_labels.clone()
                    
                else:
                    self.digit_data = torch.cat((self.digit_data,this_digit_data),0)
                    self.digit_labels = torch.cat((self.digit_labels,this_digit_labels),0)
                    
        
        
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       
        img, target = self.digit_data[index], self.digit_labels[index]
        img = Image.fromarray(img.numpy(), mode="L")

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        img = img.view(-1, 28, 28) 

        return img, target

    def __len__(self):
    
        return(self.digit_labels.size()[0])  # 總data數量
 
def Mnist_Split_Transform(root = './data'):
   
    class_split = []

    # 5類算一個task(class50~99)
    for i in range(0, 10, 2):
        class_split.append([j for j in range(i, i+2)])
    print("class split in task : ",class_split)


    mean = 33.3184
    std = 78.5675

    transform_train = transforms.Compose([
        
        transforms.ToTensor(),
        # transforms.Normalize(mean = mean/255, std=std/255)             
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        # transforms.Normalize(mean = mean/255, std=std/255)
    ])


    task = 1
    for digits in class_split:           
        # 使用label的真值(0-99作為label)
        dsets = {}
        dsets['train_true'] =  Mnist_Split(root, train=True, download=True,
                                           transform=transform_train,
                                           digits=digits)
        
        dsets['val_true'] =  Mnist_Split(root, train=False,
                                        transform=transform_test,
                                        digits=digits)
                          
        dlabel=str(task)
        
        #建立資料夾 存放各task的資料
        if not os.path.isdir('data/Mnist_split'):
            os.mkdir('data/Mnist_split')
        #task data 的 檔案名 split(1~11)_dataset.pth.tar
        torch.save(dsets,'data/Mnist_split/split'+dlabel+'_dataset.pth.tar')
        task += 1

'''Basic Cifar100 Dataset <Class> '''  
class CIFAR_Split(datasets.CIFAR100):   

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,digits=[1,2], real_label = True):
        super(CIFAR_Split, self).__init__(root, train, transform, target_transform, download)
        
        
        #print(self.train_labels.size())
        #print(self.train_data.size())

        #get only the two digits
        self.digit_labels=None
        self.digit_data=None
        self.classes= digits 
        if self.train:
            
            #loop over the given digits and extract there corresponding data
            for digit in digits:
            
                digit_mask=torch.eq(self.train_labels , digit) # 判斷train_label中的類別是否對應指定的digit，是=True, 不是=False
                digit_index=torch.nonzero(digit_mask)  # 取出digit_mask=true的label的索引值(指定labe對應的索引直)
                digit_index=digit_index.view(-1)    # 轉成一維tensor(digit_index原本就是一維陣列)
                this_digit_data=self.train_data[digit_index]    # 取出指定的data
                this_digit_labels=self.train_labels[digit_mask]    # 取出指定的label值(創造出足夠的sample空間)
                
                # training data要重新定義label
                if not real_label:
                    this_digit_labels.fill_(digits.index(digit))    # 重新定義label值(從0開始)
                if self.digit_data is None:
                    self.digit_data=this_digit_data.clone()    # 複製張量(梯度狀態一樣複製但數值無法流入)
                    self.digit_labels=this_digit_labels.clone()
                else:#concate data after first assign with first digit data
                    self.digit_data=torch.cat((self.digit_data,this_digit_data),0)
                    self.digit_labels=torch.cat((self.digit_labels,this_digit_labels),0)
               

            #self.train_data, self.train_labels = torch.load(
                #os.path.join(root, self.processed_folder, self.training_file))
        else:
                       #loop over the given digits and extract there corresponding data
            for digit in digits:
                digit_mask=torch.eq(self.test_labels , digit)    # 比較兩tensor是否相等
                digit_index=torch.nonzero(digit_mask)    # 非0元素的索引值(由左~右, 上~下讀取一個陣列)
                digit_index=digit_index.view(-1)
                this_digit_data=self.test_data[digit_index]
                this_digit_labels=self.test_labels[digit_mask]
                
                # 創建不為真值的label(label在每個task都重新從0開始標籤)
                if not real_label:
                    this_digit_labels.fill_(digits.index(digit))
                if self.digit_data is None:
                    self.digit_data=this_digit_data.clone()
                    self.digit_labels=this_digit_labels.clone()
                    
                else:
                    self.digit_data=torch.cat((self.digit_data,this_digit_data),0)
                    self.digit_labels=torch.cat((self.digit_labels,this_digit_labels),0)
                    
        
        
    
    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
       
        img, target = self.digit_data[index], self.digit_labels[index]
       

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img.numpy(), mode='RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        
        # img=img.view(-1, 3, 32, 32) # 轉成一維tensor(直接丟入全連層)
        img=img.view(-1, 3, 224, 224) 

        return img, target

    def __len__(self):
        return(self.digit_labels.size()[0])  # 總data數量
 
'''Transform and Split Cifar100 and save as file''' 
# need Declaration of <Class> CIFAR_Split
def Cifar_Split_Transform(root = './data'):
   

    # 將Cifar100 dataset切成11個任務的資料儲存(放進loader)
    # 建立TASK1(class 0,49)~TASK11(class 95,99)
    class_split = [[i for i in range(50)]]

    # 5類算一個task(class50~99)
    for i in range(50, 100, 5):
        class_split.append([j for j in range(i, i+5)])
    print("class split in task : ",class_split)


    transform_train = transforms.Compose([
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip()
        
        # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224)),
        
        #transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]) 
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    task = 1
    for digits in class_split:           
        # 使用label的真值(0-99作為label)
        dsets = {}
        dsets['train_true']=  CIFAR_Split(root, train=True, download=True,
                           transform=transform_train,
                digits=digits)
        
        dsets['val_true']=  CIFAR_Split(root, train=False,
                            transform=transform_test,
                           digits=digits)
              
        dlabel=str(task)
        
        #建立資料夾 存放各task的資料
        if not os.path.isdir('data/Cifar100_split'):
            os.mkdir('data/Cifar100_split')
        #task data 的 檔案名 split(1~11)_dataset.pth.tar
        torch.save(dsets,'data/Cifar100_split/split'+dlabel+'_dataset.pth.tar')
        task += 1

'''Basic Cifar100 Feature Dataset <Class> '''              
class Cifar_Feature_Split(Dataset):

    def __init__(self, dataset):
        super(Cifar_Feature_Split, self).__init__()
    
 
        train_loader = DataLoader(dataset, batch_size = 128, shuffle = False)
        
        FE = Resnet50_FeatureExtrator()
        FE.to(device)
        FE.eval()
        
        feature_list = [] 
        label_list =[]
        for (img, label) in train_loader:
            
            img = img.view(-1,3,224,224)
            img = img.to(device)
        
            feature = FE(img)
            feature = feature.view(-1,2048)
            feature_list.append(feature.cpu())
            label_list.append(label.cpu())
        
        self.stack_feature = torch.cat(feature_list, dim = 0)
        self.stack_label = torch.cat(label_list, dim = 0)
        feature_list.clear()
        label_list.clear()
        
    
    def __getitem__(self, index):
    
        f = self.stack_feature[index]
        label = self.stack_label[index]
        
        return f, label
        
    def __len__(self):
    
        return self.stack_label.shape[0]

'''Transform and Split Cifar100 Features and save as file''' 
# need Declaration of <Class> Cifar_Feature_Split      
def Cifar_Split_Transform_Feature(root = './data'):
   

    # 將Cifar100 dataset切成11個任務的資料儲存(放進loader)
    # 建立TASK1(class 0,49)~TASK11(class 95,99)
    class_split = [[i for i in range(50)]]

    # 5類算一個task(class50~99)
    for i in range(50, 100, 5):
        class_split.append([j for j in range(i, i+5)])
    print("class split in task : ",class_split)

    # 將Cifar100 dataset切成10個任務的資料儲存(放進loader)
    # 10類算一個task(class0~99)
    # class_split =[]                                                                                     
    # for i in range(0, 100, 10):
        # class_split.append([j for j in range(i, i+10)])
    # print("class split in task : ",class_split)
    
  

    transform_train = transforms.Compose([
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip()
        
        # transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])
        
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224)),
        
        #transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276]) 
        #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    task = 1
    for digits in class_split:           
        # 使用label的真值(0-99作為label)
        dsets = {}
        train_set =  CIFAR_Split(root, train=True, download=True,
                                    transform=transform_train,
                                    digits=digits)
                                    
        dsets['train_true'] = Cifar_Feature_Split(train_set)
        
        test_set = CIFAR_Split(root, train=False,
                            transform=transform_test,
                           digits=digits)
                           
        dsets['val_true'] = Cifar_Feature_Split(test_set)
              
        dlabel=str(task)
        
        #建立資料夾 存放各task的資料
        branch = '/50_5by10'
        branch_subset = '/feature_set'
        data_folder = './data/Cifar100_split'
        dataset_folder = data_folder + branch + branch_subset
        
        if not os.path.isdir(dataset_folder):
            if not os.path.isdir(data_folder + branch):
                os.mkdir(data_folder + branch)
            os.mkdir(dataset_folder)
        #task data 的 檔案名 split(1~11)_dataset.pth.tar
        torch.save(dsets, dataset_folder+'/split'+dlabel+'_dataset.pth.tar')
        task += 1

# Center-based Exempalar Method          
def Construct_Center_ExemplarSet(feature_set, m):
        """Construct an exemplar set for image set
        Args:
            images: np.array containing images of a class
        """
        # Compute and cache features for each example
        

        class_mean = torch.mean(feature_set, dim = 0)
        # class_mean = class_mean / np.linalg.norm(class_mean) # Normalize

        # exemplar_set = []
        exemplar_features = [] # list of Variables of shape (feature_size,)
        for k in range(m):
            
            if exemplar_features == []:
                sum_of_exemplars = torch.zeros(feature_set.shape[1])
            else:
                sum_of_exemplars = torch.sum(torch.cat(exemplar_features, dim =0), dim=0)
            
            # phi = features
            # mu = class_mean
            
            mean_of_candidate_set = 1.0/(k+1) * (feature_set + sum_of_exemplars)
            # mean_of_candidate_set = mean_of_candidate_set / np.linalg.norm(mean_of_candidate_set)# Normalize
            
            i = torch.argmin(torch.sqrt(torch.sum((class_mean - mean_of_candidate_set) ** 2, dim=1)))

  
            exemplar_features.append(feature_set[i])

        
        return torch.stack(exemplar_features,dim =0)

# Random-based Exempalar Method 
def Construct_Rand_ExemplarSet(feature_set, m):
        
        #fenerate idxs list
        fidxs = [i for i in range(feature_set.shape[0])]

        #shuffle indexs
        random.shuffle(fidxs)
        
        #sample from shffled list
        exemplar_features = [] # list of Variables of shape (feature_size,)
        for k in range(m):
            exemplar_features.append(feature_set[fidxs[k],...])

        return torch.stack(exemplar_features,dim =0)
        
'''Basic Exemplar Feature Set <Class>'''
# need to invoke <function> Construct_Center_ExemplarSet /  <function> Construct_Rand_ExemplarSet
class Exemplar_Feature_Set(Dataset):

    def __init__(self, dataset , num_exemplar, mode = "rand", label_format = "origin"):
        super(Exemplar_Feature_Set, self).__init__()
    
        data_loader = DataLoader(dataset, batch_size = 512, shuffle = False)
        
        #Store Data per each class
        class_feature_dict = {}
        for features, label in data_loader:
            
            label = label.view(-1)
            uni_label =  torch.unique(label)
   
            for l in uni_label:
                if str(l.item()) not in class_feature_dict.keys():
                    idxs = torch.argwhere(label==l)[:,0]
                    class_feature_dict.update({str(l.item()):[features[idxs,:]]})

                else:
                    idxs = torch.argwhere(label==l)[:,0]
                    class_feature_dict[str(l.item())].append(features[idxs,:])        
        
        #Sample Data from data of each class
        m = num_exemplar
        for key in class_feature_dict.keys():
                                      
            if mode == "rand":
                class_feature_dict[key] = Construct_Rand_ExemplarSet(torch.cat(class_feature_dict[key], dim = 0), m)  
            elif mode == "center":                
                class_feature_dict[key] = Construct_Center_ExemplarSet(torch.cat(class_feature_dict[key], dim = 0), m)  
        
        #Re-concate data and target from Exemplar
        self.stack_exemplar_feature = []
        self.stack_exemplar_label = []
        for key in class_feature_dict.keys():

            self.stack_exemplar_feature.append(class_feature_dict[key])
            
            if label_format == "origin":
                self.stack_exemplar_label.append(torch.ones(class_feature_dict[key].shape[0])*int(key)) 
            elif label_format == "zeros":
                self.stack_exemplar_label.append(torch.zeros(class_feature_dict[key].shape[0])) 
                
        del class_feature_dict
        
        self.stack_exemplar_feature =  torch.cat(self.stack_exemplar_feature, dim = 0)
        self.stack_exemplar_label =   torch.cat(self.stack_exemplar_label, dim = 0)
    
    def __getitem__(self, index):
    
        f = self.stack_exemplar_feature[index]
        label = self.stack_exemplar_label[index]
        
        return f, label
        
    def __len__(self):
    
        return self.stack_exemplar_label.shape[0]

class Exemplar_Feature_Combine_Set(Dataset):

    def __init__(self, old_dataset, new_dataset, origin_classes):
        super(Exemplar_Feature_Combine_Set, self).__init__()
    
        data_loader1 = DataLoader(old_dataset, batch_size = 512, shuffle = False)
        data_loader2 = DataLoader(new_dataset, batch_size = 512, shuffle = False)
        # dataloaders = [data_loader1, data_loader2]
        
        class_feature_dict = {}
        self.stack_exemplar_feature = []
        self.stack_exemplar_label  = []
        # for data_loader in dataloaders:
        for features, labels in data_loader1:
            # print(features.shape)
            self.stack_exemplar_feature.append(features)
            self.stack_exemplar_label.append(torch.zeros(labels.shape))
        
        old_data_size = torch.cat(self.stack_exemplar_label, dim = 0).shape[0]
        print("old data num: ", old_data_size)
        
        for features, labels in data_loader2:
            self.stack_exemplar_feature.append(features)
            self.stack_exemplar_label.append(labels - origin_classes + 1)
        
        total_data_size = torch.cat(self.stack_exemplar_label, dim = 0).shape[0]
        print("new data num: ",total_data_size-old_data_size)      
  
        self.stack_exemplar_feature =  torch.cat(self.stack_exemplar_feature, dim = 0)
        self.stack_exemplar_label =   torch.cat(self.stack_exemplar_label, dim = 0)
    
    def __getitem__(self, index):
    
        f = self.stack_exemplar_feature[index]
        label = self.stack_exemplar_label[index]
        
        return f, label
        
    def __len__(self):
    
        return self.stack_exemplar_label.shape[0]

class Exemplar_Feature_OrigionLabel_Combine_Set(Dataset):

    def __init__(self, old_dataset, new_dataset, origin_classes):
        super(Exemplar_Feature_OrigionLabel_Combine_Set, self).__init__()
    
        data_loader1 = DataLoader(old_dataset, batch_size = 512, shuffle = False)
        data_loader2 = DataLoader(new_dataset, batch_size = 512, shuffle = False)
        # dataloaders = [data_loader1, data_loader2]
        
        class_feature_dict = {}
        self.stack_exemplar_feature = []
        self.stack_exemplar_label  = []
        # for data_loader in dataloaders:
        for features, labels in data_loader1:
            # print(features.shape)
            self.stack_exemplar_feature.append(features)
            self.stack_exemplar_label.append(labels)
        
        old_data_size = torch.cat(self.stack_exemplar_label, dim = 0).shape[0]
        print("old data num: ", old_data_size)
        
        for features, labels in data_loader2:
            self.stack_exemplar_feature.append(features)
            self.stack_exemplar_label.append(labels)
        
        total_data_size = torch.cat(self.stack_exemplar_label, dim = 0).shape[0]
        print("new data num: ",total_data_size-old_data_size)      
  
        self.stack_exemplar_feature =  torch.cat(self.stack_exemplar_feature, dim = 0)
        self.stack_exemplar_label =   torch.cat(self.stack_exemplar_label, dim = 0)
    
    def __getitem__(self, index):
    
        f = self.stack_exemplar_feature[index]
        label = self.stack_exemplar_label[index]
        
        return f, label
        
    def __len__(self):
    
        return self.stack_exemplar_label.shape[0]

class Shift_Label_Dataset(Dataset):
    
    def __init__(self, origin_dataset, origin_class_logit_range = [51,55], target_class_logit_range = [1,5]):
        super(Shift_Label_Dataset, self).__init__()
        
        self.stack_feature = origin_dataset.stack_feature
        
        init_shift = target_class_logit_range[0] - origin_class_logit_range[0]
        end_shift = target_class_logit_range[1] - origin_class_logit_range[1]
        
        if init_shift != end_shift:
            raise "shifted range {} not match origin range {}".format(origin_class_logit_range, target_class_logit_range)
        
        else:
            self.stack_label = origin_dataset.stack_label + init_shift
        
    def __getitem__(self, index):
    
        f = self.stack_feature[index]
        l = self.stack_label[index]
        
        return f, l
        
    def __len__(self):
    
        return stack_label[index].shape[0]
        
def MeanCovCompute(sorted_train_loader, task , TRAIN_SAMPLE, TRAIN_CLASS, class_range = [0,50],  data_dir = "./PseudoExemplar"):
    
    record_root = './record'
    # model_dict = torch.load('{}/task{}_vae_model.pth.tar'.format(record_root, task))

    model = CVAE(28*28, cfg.z_dim, 10).to(device)  
    
    model_file = record_root + '/task{}_cvae_model.pth.tar'.format(task)
    model_dict = torch.load(model_file)
    model.load_state_dict(model_dict)
    
    # insert_chect = cfg.insert_check_dict
    # ss_size = cfg.ss_sizes_dict    
    
    # AE = SSCVAE(base, insert_chect, ss_size).to(device) 
    
    # if task != 1:
        # ss_file =  record_root + '/task{}_encoder_ss.pth.tar'.format(task)
        # ss_dict = torch.load(ss_file)
        # AE.EncoderSS1.load_state_dict(ss_dict)
        
        # ss_file =  record_root + '/task{}_decoder_ss.pth.tar'.format(task)
        # ss_dict = torch.load(ss_file)
        # AE.DecoderSS1.load_state_dict(ss_dict)    
    AE = model
    AE = AE.eval()
    AE.to(device)
    
    prototype = cfg.z_dim
    feature_size = 784
    

   #TRAIN_CLASS

    # print('Train_samples: {}'.format(TRAIN_SAMPLE))
    # print('Train_classes: {}'.format(TRAIN_CLASS))

    with torch.no_grad():
        
        all_features = torch.empty(TRAIN_SAMPLE, prototype)
        all_labels = torch.empty(TRAIN_SAMPLE, )
        
        accu_data = 0
        data_per_class = torch.zeros(TRAIN_CLASS)
        
        for step, (x, b_label) in enumerate(sorted_train_loader):
            
            # torch.cuda.empty_cache()
            
            x = x.view(-1, feature_size).to(device)
            b_label = b_label.to(torch.int64)
            
            # _, _, z_code, _ = AE(x)
            onehot = one_hot(b_label, 10)
            mu, var = AE.encode(x,onehot)
            z_code = AE.reparameterize(mu, var)
         
            for j in range(z_code.size()[0]):

                all_features[accu_data + j] = z_code[j]
                all_labels[accu_data + j] = b_label[j]

                data_per_class[b_label[j]] += 1
                
            accu_data += z_code.size()[0]
                

    # 把訓練資料按照類別排好
    # sort_feature, sort_label = sort_data(all_features, all_labels)
    sort_feature = all_features
    sort_label = all_labels
    
    
    mean_feature = torch.zeros([TRAIN_CLASS, prototype])
    # cov_feature = torch.zeros([TRAIN_CLASS, prototype, prototype])
    cov_feature = torch.zeros([TRAIN_CLASS,  prototype])
    

    init = 0
    for i in range(TRAIN_CLASS):
        
        
        des = int(init + data_per_class[i])
        
        # print(sort_feature[init:des].shape)
        # print(np.mean(np.array([0,1,2,3])))
        if i >= class_range[0] and i<class_range[1]:
            # print(sort_feature[init:des])
            mean_feature[i] = torch.mean(sort_feature[init:des], dim = 0)
            # cov_feature[i] = torch.cov(sort_feature[init:des].transpose(0,1))
            cov_feature[i] = torch.var(sort_feature[init:des], dim = 0)

        init += int(data_per_class[i])
    
        np.set_printoptions(threshold=np.inf)
 
 
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        
    if not os.path.isdir(data_dir + "/MeanCov"):
        os.mkdir(data_dir + "/MeanCov") 
        
    task_class_mean_file = data_dir + "/MeanCov" +'/task{}_feature_mean.pth.tar'.format(task)
    task_class_cov_file = data_dir + "/MeanCov" +'/task{}_feature_cov.pth.tar'.format(task)
    
    torch.save(mean_feature[class_range[0]:class_range[1]], task_class_mean_file)
    torch.save(cov_feature[class_range[0]:class_range[1]], task_class_cov_file)
    # print(mean_feature[class_range[0]:class_range[1]].shape)
    # sys.exit()

class PseudoExemplar(Dataset): 
 
    def __init__(self, sample_per_class, class_digits, task):
        super(PseudoExemplar, self).__init__()
        '''
        class_digits = list [0,49] 
        '''

        all_mean = torch.load('./PseudoExemplar/MeanCov/task{}_feature_mean.pth.tar'.format(task))
        all_std = torch.load('./PseudoExemplar/MeanCov/task{}_feature_cov.pth.tar'.format(task))
        
        

        AE = VAE()        
        record_root = './record'
        model_dict = torch.load('{}/task{}_vae_model.pth.tar'.format(record_root, task))   
        AE.load_state_dict(model_dict)
        AE = AE.eval()
        AE.to(device)

        class_digits_list = [ i for i in range(class_digits[0], class_digits[1]+1)]
        class_num = len(class_digits_list)
        # print(class_digits_list)

        all_data_list = []
        all_label_list = []             
        for i, c in enumerate(class_digits_list):
            
            samples = sample_per_class #int(total_sample / class_num)
            mean = all_mean[i,...]
            std  = all_std[i,...]

            #Produce prototype from old class mean and std
            
            # std = std * torch.eye(std.shape[0])
            # n = MultivariateNormal(mean,std)
            # sys.exit()
                       
            # instance = n.sample((samples, ))#.float()
            
            std = torch.exp(0.5 * std)

            instance = std * torch.randn([samples, std.shape[0]]) + mean

            instance = instance.to(device).float()
            # print(instance.shape)

            #decode prototype to the feature size 
            instance = AE.decode(instance)
            instance = instance.detach()
            instance = instance.view(-1,2048)

            labels = torch.zeros( int(sample_per_class) ) + c
            
            all_data_list.append(instance.cpu())             
            all_label_list.append(labels.cpu())
            
            # self.all_data = torch.cat((self.all_data, instance), 0)
            # self.all_labels = torch.cat((self.all_labels, labels), 0)

            torch.set_printoptions(threshold=np.inf)
        
        self.all_data = torch.cat(all_data_list, dim = 0)
        self.all_labels = torch.cat(all_label_list, dim = 0)
        
        all_data_list.clear()
        all_label_list.clear()
        # print(class_digits_list)
        # print(self.all_data.shape)
        # print(self.all_labels.shape)
        
    
    def __getitem__(self, index):
    
        prototype = self.all_data[index]
        target = self.all_labels[index]
        

        return prototype, target

    def __len__(self):
    
        return(self.all_labels.size()[0])  

class CVAE_PseudoExempar1(Dataset):
    
    def __init__(self, sample_per_class, class_digits, task):
        super(CVAE_PseudoExempar, self).__init__()
        '''
        class_digits = list [0,49] 
        '''
        all_mean = torch.load('./PseudoExemplar/MeanCov/task{}_feature_mean.pth.tar'.format(task))
        all_std = torch.load('./PseudoExemplar/MeanCov/task{}_feature_cov.pth.tar'.format(task))
        
        record_root = './record'        
        
        model = CVAE(28*28, cfg.z_dim, 10).to(device)  
        
        model_file = record_root + '/task{}_cvae_model.pth.tar'.format(task)
        model_dict = torch.load(model_file)
        model.load_state_dict(model_dict)
        
        # insert_chect = cfg.insert_check_dict
        # ss_size = cfg.ss_sizes_dict    
        
        # AE = SSCVAE(base, insert_chect, ss_size).to(device) 
        
        # if task != 1:
            # ss_file =  record_root + '/task{}_encoder_ss.pth.tar'.format(task)
            # ss_dict = torch.load(ss_file)
            # AE.EncoderSS1.load_state_dict(ss_dict)
            
            # ss_file =  record_root + '/task{}_decoder_ss.pth.tar'.format(task)
            # ss_dict = torch.load(ss_file)
            # AE.DecoderSS1.load_state_dict(ss_dict)    
            
        # AE = CVAE( feature_size = 784 , latent_size = 50, class_size = 10)
        AE = model
        AE = AE.eval()
        AE.to(device)

        class_digits_list = [ i for i in range(class_digits[0], class_digits[1]+1)]
        class_num = len(class_digits_list)
        # print(class_digits_list)

        all_data_list = []
        all_label_list = []             
        for i, c in enumerate(class_digits_list):
            
            samples = sample_per_class #int(total_sample / class_num)
            mean = all_mean[i,...]
            std  = all_std[i,...]

            #Produce prototype from old class mean and std
            
            # std = std * torch.eye(std.shape[0])
            # n = MultivariateNormal(mean,std)
            # sys.exit()
                       
            # instance = n.sample((samples, ))#.float()

        
            # std = torch.exp(0.5 * std)

            instance = std * torch.randn([samples, std.shape[0]]) + mean

            instance = instance.to(device).float()
            # print(instance.shape)

            #decode prototype to the feature size 
            labels = torch.zeros( int(sample_per_class) ) + c
            labels = labels.to(torch.int64)
            onehot = one_hot(labels, 10)
            instance = AE.decode(instance, onehot)
            instance = instance.detach()
            instance = instance.view(-1,784)


            all_data_list.append(instance.cpu())             
            all_label_list.append(labels.cpu())
            
            # self.all_data = torch.cat((self.all_data, instance), 0)
            # self.all_labels = torch.cat((self.all_labels, labels), 0)

            torch.set_printoptions(threshold=np.inf)
        
        self.all_data = torch.cat(all_data_list, dim = 0)
        self.all_labels = torch.cat(all_label_list, dim = 0)
        
        all_data_list.clear()
        all_label_list.clear()
        # print(class_digits_list)
        # print(self.all_data.shape)
        # print(self.all_labels.shape)
        
    
    def __getitem__(self, index):
    
        prototype = self.all_data[index]
        target = self.all_labels[index]
        

        return prototype.view(-1,28,28), target

    def __len__(self):
    
        return(self.all_labels.size()[0])  

class CVAE_PseudoExempar(Dataset):
    
    def __init__(self, sample_per_class, class_digits, task):
        super(CVAE_PseudoExempar, self).__init__()
        '''
        class_digits = list [0,49] 
        '''
        # all_mean = torch.load('./PseudoExemplar/MeanCov/task{}_feature_mean.pth.tar'.format(task))
        # all_std = torch.load('./PseudoExemplar/MeanCov/task{}_feature_cov.pth.tar'.format(task))
        
        record_root = cfg.record_root     
        
        cur_class  = cfg.base_classes + cfg.incre_classes * (task - 1)
        feature_size = cfg.input_dim
        model = CVAE(cfg.input_dim, cfg.z_dim, cur_class).to(device)  
        
        model_file = record_root + '/task{}_cvae_model.pth.tar'.format(task)
        model_dict = torch.load(model_file)
        model.load_state_dict(model_dict)

        AE = model
        AE = AE.eval()
        AE.to(device)

        class_digits_list = [ i for i in range(class_digits[0], class_digits[1]+1)]
        class_num = len(class_digits_list)
        # print(class_digits_list)

        all_data_list = []
        all_label_list = []             
        for i, c in enumerate(class_digits_list):
            
            samples = sample_per_class #int(total_sample / class_num)
            # mean = all_mean[i,...]
            # std  = all_std[i,...]

            #Produce prototype from old class mean and std
            
            # std = std * torch.eye(std.shape[0])
            # n = MultivariateNormal(mean,std)
            # sys.exit()                       
            # instance = n.sample((samples, ))#.float()
            # std = torch.exp(0.5 * std)
            # instance = std * torch.randn([samples, std.shape[0]]) + mean
            instance = torch.randn([samples, cfg.z_dim]) 

            instance = instance.to(device).float()
            # print(instance.shape)

            #decode prototype to the feature size 
            labels = torch.zeros( int(sample_per_class) ) + c
            labels = labels.to(torch.int64)
            onehot = one_hot(labels, cur_class)
            instance = AE.decode(instance, onehot)
            instance = instance.detach()
            instance = instance.view(-1,feature_size)


            all_data_list.append(instance.cpu())             
            all_label_list.append(labels.cpu())
            
            # self.all_data = torch.cat((self.all_data, instance), 0)
            # self.all_labels = torch.cat((self.all_labels, labels), 0)

            torch.set_printoptions(threshold=np.inf)
        
        self.all_data = torch.cat(all_data_list, dim = 0)
        self.all_labels = torch.cat(all_label_list, dim = 0)
        
        all_data_list.clear()
        all_label_list.clear()
        # print(class_digits_list)
        # print(self.all_data.shape)
        # print(self.all_labels.shape)
        
    
    def __getitem__(self, index):
    
        prototype = self.all_data[index]
        target = self.all_labels[index]
        

        return prototype.view(cfg.dataloader_dim), target

    def __len__(self):
    
        return(self.all_labels.size()[0])  


if __name__ == "__main__":
    Mnist_Split_Transform()