#config
import os

# code_root = './MSS_Basic'
code_root = './MSS_Basic_10by10_50'

class Config():

    def __init__(self):
        
        # self.name = "CVAE_Basic10"
        
        # class split setting
        self.base_classes = 10
        self.incre_classes = 10
        self.final_classes = 100
        
        # dataset path and select branch
        self.data_root = './data'
        self.data_branch = '/10_10/feature_set'
        self.data_name = '/Cifar100_Split'
        self.record_root = code_root + '/record'
        self.code_root = code_root
        # self.recover_folder = self.record_root + '/RecovResult'
        # self.generate_folder = self.record_root + '/GenerResult'
        
        self.feature_dim = 2048
        self.input_dim = 2048
        # self.z_dim = 100
        self.data_size = [1, 2048] #[1,28,28]
        self.dataloader_dim = 2048
        
        # record name
        self.save_cnn_name = "cnn_model"
        self.save_ss_name = "ss_set"
        # self.save_cvae_name = "cvae_model"
        self.base_cnn_train_his_name = "base_cnn_train"
        # self.base_cvae_train_his_name = "base_cvae_train"

        self.icl_cnn_train_his_name = "icl_cnn_train"
        # self.icl_cvae_train_his_name = "icl_cvae_train"
                
        # Sampling hyper-parameter  
        self.num_exemplar = 50       
        
        # common hyper-parameter 
        self.batch_size = 512 

        # Base training parameter        
        # CNN
        self.base_cnn_epochs = 15
        self.base_cnn_lr = 1e-3
        self.base_cnn_wd = 0.0005
        
        # CVAE
        # self.base_cvae_epochs = 100
        # self.base_cvae_lr = 1e-3    
        
        # Base training parameter
        # CNN      
        self.icl_cnn_epoch = 100
        self.icl_cnn_lr_dict = {"base":5e-3, "ss":5e-4}  
        
        # CVAE
        # self.icl_cvae_lr = 1e-3
        # self.icl_cvae_epochs = 30

        # cnn ss parameter
        self.insert_check_list = [1,1,1] # SS used check
        self.ss_sizes = [2048, 1024, 1024] # SS size at each layer

        # loss ratio
        # self.init_ratio = 1e-4
        # self.incre_ratio = 0.001
        # self.half_cycle = 8
        # self.base_wd = 0.0005 #weight decay
        
        # self.icl_epochs = 100
        # self.icl_LastFC_lr =  5e-4 # last FC learning Rate
        # self.icl_SS_lr =  5e-3
        # self.icl_wd = 0.0005 # weight decay 

        
        # self.insert_check_dict = {  "en" : {"share" : [1,1],
                                            # "mean" : [1],
                                            # "var" : [1] },
                                            
                                    # "de" : [1,1,1] }
                                    
        # self.ss_sizes_dict = {  "en" : {    "share" : [784+10, 400],
                                            # "mean" : [400,  self.z_dim],
                                            # "var" : [400,  self.z_dim ]},
                                    
                            # "de" : [ self.z_dim + 10, 400, 400] }
        '''assumption : ratio down while # current classes / # acculmate classes down'''
        # self.loss_ratio = [0.1, 0.09, 0.08, 0.077, 0.071, 0.066, 0.0625, 0.058, 0.055, 0.05] # for use KD only
        # self.loss_ratio = [w * 4 for w in ce_w ] # for use KL only
        # self.loss_ratio = [0.3, 0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # for fixKD 
        # self.loss_ratio = [ 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.35, 0.25, 0.25, 0.25] # for fixKL
        # self.loss_ratio = [w * 5 for w in ce_w ]# for fixKL        

        # test setting
        self.test_tasks = 11               