from torchvision import datasets, models, transforms
import torchvision
import torch.nn as nn
import torch.utils.data as data
from torch.utils.data import Dataset, DataLoader
import torch

from PIL import Image
import numpy as np
import sys
import os
import random

from ModelCollection import *
from SupportFunction import *
from Config import *


device = "cuda" if torch.cuda.is_available() else "cpu"

'''Basic Cifar100 Dataset <Class> '''  
class CIFAR_Split(datasets.CIFAR100):   

    def __init__(self, root, train=True, transform=None, target_transform=None, download=False,digits=[1,2], real_label = True):
        super(CIFAR_Split, self).__init__(root, train, transform, target_transform, download)

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
def Cifar_Split_Transform(base_classes = 50 , increm_classes = 5, total_classes = 100, data_root = './data'):
   
    # task1 class split digits [0 ~ base_classes-1]
    class_split = [[i for i in range(base_classes)]]

    # task 2 ~ n class split digits [base_classes + increm_classes(i) ~ base_classes + increm_classes(i+1)] 
    for i in range(base_classes, total_classes, increm_classes):
        class_split.append([j for j in range(i, i+5)])
    print("class split in task : ",class_split)

    # dataset transform Cifar specific hyparameter
    transform_train = transforms.Compose([
        
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224)),
        transforms.RandomCrop(224, padding=4),
        transforms.RandomHorizontalFlip()
    ])

    # Normalize test set same as training set without augmentation
    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.Resize(size=(224,224)),
    ])


    task = 1
    for digits in class_split:           
        dsets = {}
        # instance of Cifar Split Dataset
        dsets['train_true']=  CIFAR_Split(data_root, 
                                          train=True, 
                                          download=True,
                                          transform=transform_train,
                                          digits=digits)
        
        dsets['val_true']=  CIFAR_Split(data_root, 
                                        train=False,
                                        transform=transform_test,
                                        digits=digits)
              
        dlabel=str(task) # specify dataset
        
        #save dataset 
        if not os.path.isdir(data_root + '/Cifar100_split'):
            os.mkdir(data_root + '/Cifar100_split')
            
        torch.save(dsets, data_root +'/Cifar100_split/split'+ dlabel +'_dataset.pth.tar')
        task += 1

'''Basic Cifar100 Feature Dataset <Class> '''              
class Cifar_Feature_Split(Dataset):

    def __init__(self, dataset, feature_dim, FE_mode = "res50"):
        super(Cifar_Feature_Split, self).__init__()
    
        train_loader = DataLoader(dataset, batch_size = 128, shuffle = False)
        
        # FE = Resnet50_FeatureExtrator()
        FE = FeatureExtractor(mode = FE_mode)
        
        FE.to(device)
        FE.eval()
        
        feature_list = [] 
        label_list =[]
        for (img, label) in train_loader:
            
            img = img.view(-1,3,224,224)
            img = img.to(device)
        
            feature = FE(img)
            feature = feature.view(-1,feature_dim)
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
def Cifar_Split_Transform_Feature(base_classes = 50 , increm_classes = 5, total_classes = 100, data_root = './data', branch = '/50_5by10'):
   

    # task1 class split digits [0 ~ base_classes-1]
    class_split = [[i for i in range(base_classes)]]

    # task 2 ~ n class split digits [base_classes + increm_classes(i) ~ base_classes + increm_classes(i+1)] 
    for i in range(base_classes, total_classes, increm_classes):
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
    ])


    task = 1
    for digits in class_split:           
        # 使用label的真值(0-99作為label)
        dsets = {}
        train_set =  CIFAR_Split(   data_root, 
                                    train=True, 
                                    download=True,
                                    transform=transform_train,
                                    digits=digits)
                                    
        dsets['train_true'] = Cifar_Feature_Split(train_set)
        
        test_set = CIFAR_Split( data_root, 
                                train=False,
                                transform=transform_test,
                                digits=digits)
                           
        dsets['val_true'] = Cifar_Feature_Split(test_set)
              
        dlabel=str(task)
        
        #建立資料夾 存放各task的資料
        # branch = '/50_5by10'
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
                class_feature_dict[key] = self.Construct_Rand_ExemplarSet(torch.cat(class_feature_dict[key], dim = 0), m)  
            elif mode == "center":                
                class_feature_dict[key] = self.Construct_Center_ExemplarSet(torch.cat(class_feature_dict[key], dim = 0), m)  
        
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

    # Center-based Exempalar Method          
    def Construct_Center_ExemplarSet(self, feature_set, m):
        """Construct an exemplar set for image set
        Args:
            images: np.array containing images of a class
        """
        # Compute and cache features for each example
        class_mean = torch.mean(feature_set, dim = 0)
        
        exemplar_features = [] # list of Variables of shape (feature_size,)
        for k in range(m):
            
            if exemplar_features == []:
                sum_of_exemplars = torch.zeros(feature_set.shape[1])
            else:
                sum_of_exemplars = torch.sum(torch.cat(exemplar_features, dim =0), dim=0)

            
            mean_of_candidate_set = 1.0/(k+1) * (feature_set + sum_of_exemplars)

            i = torch.argmin(torch.sqrt(torch.sum((class_mean - mean_of_candidate_set) ** 2, dim=1)))

            exemplar_features.append(feature_set[i])        
            
        return torch.stack(exemplar_features,dim =0)
    
    # Random-based Exempalar Method 
    def Construct_Rand_ExemplarSet(self, feature_set, m):
            
        #fenerate idxs list
        fidxs = [i for i in range(feature_set.shape[0])]

        #shuffle indexs
        random.shuffle(fidxs)
        
        #sample from shffled list
        exemplar_features = [] # list of Variables of shape (feature_size,)
        for k in range(m):
            exemplar_features.append(feature_set[fidxs[k],...])

        return torch.stack(exemplar_features,dim =0)

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

def SSMeanCovCompute(sorted_train_loader, task , total_sample, total_class, class_range = [0,50], model_class = 50, model_task = 1):    
    
    cfg = Config()
    feature_size = cfg.feature_dim
    data_dir = cfg.pseudo_root
    
    base_model = Load_Model(model_class, model_task, model_type = 'ae')
 
    en_ss_list1 = Load_EN_Kernel(insertion_check_list, kernel_size, task-1, training=False)
    en_ss_list2 = Load_EN_Kernel(insertion_check_list, kernel_size, task, training=False) 
                
    #conbine model and kernel
    if en_ss_list2 == None:
         SSAE = base_model
    else:
        SSAE = SSAEwithFE( base_model, insertion_check_list, kernel_size)
        SSAE.ss_set2 = en_ss_list2


    SSAE = SSAE.to(device)

    SSAE.eval()    
           
    SSAE.to(device)

    prototype = total_class #100
    # feature_size = 2048
            
    with torch.no_grad():
        
        all_features = torch.empty(total_sample, prototype)
        all_labels = torch.empty(total_sample, )
                
        # from 0 ~ current class
        data_per_class = torch.zeros(total_class) 
        accu_data = 0
        
        # iterative data from last class ~ current class
        for step, (x, labels) in enumerate(sorted_train_loader):
            
            x = x.view(-1, feature_size).to(device)
            labels = labels.to(torch.int64)            
            
            if task==1:
                z_code = SSAE.encode(x, 'f')    
            else:
                z_code = SSAE.single_encode(x, 'f')       
            
            for j in range(z_code.size()[0]):

                all_features[accu_data + j] = z_code[j, 0:total_class]
                all_labels[accu_data + j] = labels[j]
                
                # in fact, only count for last class ~ current class
                data_per_class[labels[j]] += 1
                
            accu_data += z_code.size()[0]
                

    # data is already sorted by dataloader
    sort_feature = all_features 
    sort_label = all_labels
    
    # initial size for 0 ~ current class, actually, only use last class ~ current class
    mean_feature = torch.zeros([total_class, prototype])
    cov_feature = torch.zeros([total_class, prototype, prototype])
    # cov_feature = torch.zeros([total_class,  prototype])

    init = 0
    # iterative from 0 ~ current class
    for i in range(total_class):
               
        des = int(init + data_per_class[i])
        
        #compute mean and covariance of last class ~ current class
        if i >= class_range[0] and i<class_range[1]:
        
            mean_feature[i] = torch.mean(sort_feature[init:des], dim = 0)
            cov_feature[i] = torch.cov(sort_feature[init:des].T)
            # cov_feature[i] = torch.var(sort_feature[init:des], dim = 0)

        init += int(data_per_class[i])
    
        np.set_printoptions(threshold=np.inf)
 
 
    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        
    if not os.path.isdir(data_dir + "/MeanCov"):
        os.mkdir(data_dir + "/MeanCov") 
        
    task_class_mean_file = data_dir + "/MeanCov" +'/task{}_feature_mean.pth.tar'.format(task)
    task_class_cov_file = data_dir + "/MeanCov" +'/task{}_feature_cov.pth.tar'.format(task)
    
    # save only for [last class ~ current class] i.e. [class_range[0]:class_range[1]]
    torch.save(mean_feature[class_range[0]:class_range[1]], task_class_mean_file)
    torch.save(cov_feature[class_range[0]:class_range[1]], task_class_cov_file)

class SSPseudoExemplar(Dataset): 
 
    def __init__(self, sample_per_class, class_digits, task, model_task, model_class):
        super(PseudoExemplar, self).__init__()
        '''
        class_digits = list [0,49] 
        '''
        cfg = Config()
        incre_classes = cfg.incre_classes
        record_root = cfg.record_root
        pseudo_root = cfg.pseudo_root
        de_insert_check_list = cfg.insert_check_dict["de"]
        de_ss_size_list = cfg.ss_size_dict["de"]
        feature_dim = cfg.feature_dim
        
        #load mean and variance
        all_mean = torch.load(pseudo_root+'/MeanCov/task{}_feature_mean.pth.tar'.format(task))
        all_std = torch.load(pseudo_root+'/MeanCov/task{}_feature_cov.pth.tar'.format(task))


        base_model = AutoEncoderwithFE( base_class = model_class )
        
        model_dict = torch.load('{}/task{}_model.pth.tar'.format(record_root, model_task))   
        base_model.load_state_dict(model_dict)
        
        insertion_check_list = de_insert_check_list
        de_ss_sizes = de_ss_size_list
        de_ss_list2 = Load_DE_Kernel(insertion_check_list, de_ss_sizes, task, training = False)    
        
        #conbine model and kernel
        if de_ss_list2 == None:
            
            SSAE = base_model
        else:
            SSAE = SSAEwithFE( base_model, insertion_check_list, kernel_size)
            SSAE.de_ss_set2 = de_ss_list2 
          

        SSAE = SSAE.eval()
        SSAE.to(device)

        class_digits_list = [ i for i in range(class_digits[0], class_digits[1])]
        class_num = len(class_digits_list)

        all_data_list = []
        all_label_list = []             
        for i, c in enumerate(class_digits_list):
            
            samples = sample_per_class # int(total_sample / class_num)
            mean = all_mean[i,...]
            std  = all_std[i,...]

            #Produce prototype from old class mean and std  
            std = std * torch.eye(std.shape[0])
            
            n = MultivariateNormal(mean, std)
            
            instance = n.sample((samples, ))#.float()

            instance = instance.to(device).float()
            
            # cover the new weight influence
            if task < model_task:
                zero_padding = torch.zeros(samples, incre_classes*(model_task - task)).to(device)
                instance = torch.cat([instance, zero_padding], dim = 1)
            
            #decode prototype to the feature size 
            instance = SSAE.decode(instance)
            instance = instance.detach()
            instance = instance.view(-1,feature_dim)

            labels = torch.zeros( int(sample_per_class) ) + c
            
            all_data_list.append(instance.cpu())             
            all_label_list.append(labels.cpu())


        self.all_data = torch.cat(all_data_list, dim = 0)
        self.all_labels = torch.cat(all_label_list, dim = 0)
        
        all_data_list.clear()
        all_label_list.clear()
    
    def __getitem__(self, index):
    
        prototype = self.all_data[index]
        target = self.all_labels[index]
        

        return prototype, target

    def __len__(self):
    
        return(self.all_labels.size()[0])  

def MeanCovCompute(sorted_train_loader, task , total_sample, total_class, class_range = [0,50], model_class = 50, model_task = 1):

    cfg = Config()
    feature_size = cfg.feature_dim
    data_dir = cfg.pseudo_root        
    record_root = cfg.record_root
    
    model_dict = torch.load('{}/task{}_model.pth.tar'.format(record_root, model_task))
    
    AE = AutoEncoderwithFE(model_class)    
    AE.load_state_dict(model_dict)
    AE = AE.eval()
    AE.to(device)

    prototype = model_class 
            
    with torch.no_grad():
        
        all_features = torch.empty(total_sample, prototype)
        all_labels = torch.empty(total_sample, )
                
        # from 0 ~ current class
        data_per_class = torch.zeros(total_class) 
        accu_data = 0
        
        # iterative data from last class ~ current class
        for step, (x, labels) in enumerate(sorted_train_loader):
            
            x = x.view(-1, feature_size).to(device)
            labels = labels.to(torch.int64)            

            z_code = AE.encode(x, inp_mode = 'f')

            # z_code = torch.nn.functional.sigmoid(z_code)
                     
            for j in range(z_code.size()[0]):

                all_features[accu_data + j] = z_code[j]
                all_labels[accu_data + j] = labels[j]
                
                # in fact, only count for last class ~ current class
                data_per_class[labels[j]] += 1
                
            accu_data += z_code.size()[0]

    # data is already sorted by dataloader
    sort_feature = all_features 
    sort_label = all_labels
    
    # initial size for 0 ~ current class, actually, only use last class ~ current class
    mean_feature = torch.zeros([total_class, prototype])
    cov_feature = torch.zeros([total_class, prototype, prototype])

    init = 0
    # iterative from 0 ~ current class
    for i in range(total_class):
               
        des = int(init + data_per_class[i])
        
        #compute mean and covariance of last class ~ current class
        if i >= class_range[0] and i<class_range[1]:
        
            mean_feature[i] = torch.mean(sort_feature[init:des], dim = 0)
            cov_feature[i] = torch.cov(sort_feature[init:des].T)

        init += int(data_per_class[i])
    
        np.set_printoptions(threshold=np.inf)

    if not os.path.isdir(data_dir):
        os.mkdir(data_dir)
        
    if not os.path.isdir(data_dir + "/MeanCov"):
        os.mkdir(data_dir + "/MeanCov") 
        
    task_class_mean_file = data_dir + "/MeanCov" +'/task{}_feature_mean.pth.tar'.format(task)
    task_class_cov_file = data_dir + "/MeanCov" +'/task{}_feature_cov.pth.tar'.format(task)
    
    # save only for [last class ~ current class] i.e. [class_range[0]:class_range[1]]
    torch.save(mean_feature[class_range[0]:class_range[1]], task_class_mean_file)
    torch.save(cov_feature[class_range[0]:class_range[1]], task_class_cov_file)

class PseudoExemplar(Dataset): 
 
    def __init__(self, sample_per_class, class_digits, task, model_task, model_class):
        super(PseudoExemplar, self).__init__()
        '''
        class_digits = list [0,49] 
        '''
        cfg = Config()
        incre_classes = cfg.incre_classes
        record_root = cfg.record_root
        pseudo_root = cfg.pseudo_root
        feature_dim = cfg.feature_dim

        all_mean = torch.load(pseudo_root+'/MeanCov/task{}_feature_mean.pth.tar'.format(task))
        all_std = torch.load(pseudo_root+'/MeanCov/task{}_feature_cov.pth.tar'.format(task))
        
        AE = AutoEncoderwithFE( base_class = model_class )        
        model_dict = torch.load('{}/task{}_model.pth.tar'.format(record_root, model_task))   
        AE.load_state_dict(model_dict)
        AE = AE.eval()
        AE.to(device)

        class_digits_list = [ i for i in range(class_digits[0], class_digits[1])]
        class_num = len(class_digits_list)

        all_data_list = []
        all_label_list = []             
        for i, c in enumerate(class_digits_list):
            
            samples = sample_per_class # int(total_sample / class_num)
            mean = all_mean[i,...]
            std  = all_std[i,...]

            #Produce prototype from old class mean and std  

            # std = torch.tril(std)
            std = std * torch.eye(std.shape[0])
            # print(std)
            # L = torch.linalg.cholesky(std)
            
            n = MultivariateNormal(mean, std)
            
            instance = n.sample((samples, ))#.float()

            instance = instance.to(device).float()
            # print(instance.shape)           
            
            #decode prototype to the feature size 
            instance = AE.decode(instance)
            instance = instance.detach()
            instance = instance.view(-1,feature_dim)

            labels = torch.zeros( int(sample_per_class) ) + c
            
            all_data_list.append(instance.cpu())             
            all_label_list.append(labels.cpu())

        
        self.all_data = torch.cat(all_data_list, dim = 0)
        self.all_labels = torch.cat(all_label_list, dim = 0)
        
        all_data_list.clear()
        all_label_list.clear()
    
    def __getitem__(self, index):
    
        prototype = self.all_data[index]
        target = self.all_labels[index]
        
        return prototype, target

    def __len__(self):
    
        return(self.all_labels.size()[0])  

def Create_All_Task_PseudoSet(num_exemplar, task, last_classes, mode = "ae"):
    
    cfg = Config()
    base_classes = cfg.base_classes
    increm_classes = cfg.incre_classes
    
    start_digit = 0
    end_digit = base_classes    
    PseudoSetList = []
    for tidx in range(task-1):
        print("pseudo:", tidx) 
        
        if mode == "ae":
            PseudoOldSet = PseudoExemplar(  sample_per_class = num_exemplar, 
                                            class_digits = [start_digit, end_digit],
                                            task = tidx+1,
                                            model_task = task-1,
                                            model_class = last_classes)
        if mode == "ssae":                                           
            PseudoOldSet = SSPseudoExemplar(  sample_per_class = num_exemplar, 
                                    class_digits = [start_digit, end_digit],
                                    task = tidx+1,
                                    model_task = task-1,
                                    model_class = last_classes)
                                    
        PseudoSetList.append(PseudoOldSet)
        end_digit += increm_classes
        start_digit += increm_classes * (tidx>0) + base_classes * (tidx==0)
    
    return PseudoSetList

def Compute_All_Task_MeanCov(all_dataset_list, task, last_classes, mode = "ae"):
       
    cfg = Config()
    base_classes = cfg.base_classes
    increm_classes = cfg.incre_classes
    
    start_digit = 0
    end_digit = base_classes  
    cumul_class = base_classes
    for tidx, dset in enumerate(all_dataset_list):

        sorted_train_loader = DataLoader(dset, batch_size = cfg.batch_size, shuffle = False)
        if mode =="ae":
            MeanCovCompute( sorted_train_loader,
                            task = tidx+1,
                            total_sample = len(dset),
                            total_class = end_digit, 
                            class_range = [start_digit, end_digit],
                            model_class = last_classes + increm_classes, 
                            model_task = task)
        if mode == "ssae":
            SSMeanCovCompute( sorted_train_loader,
                            task = tidx+1,
                            total_sample = len(dset),
                            total_class = end_digit, 
                            class_range = [start_digit, end_digit],
                            model_class = last_classes + increm_classes, 
                            model_task = task)        
            
                        
        cumul_class += increm_classes
        end_digit += increm_classes
        start_digit += increm_classes * (tidx>0) + base_classes * (tidx==0)
    
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
        
        cfg = Config()
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