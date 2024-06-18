#config
import os

code_root = "./Free_KD_Only_10by10"
class Config():

    def __init__(self):

        self.num_exemplar = 10
    
        self.data_root = "./data"
        self.record_root = code_root + "/record"
        self.pseudo_root = code_root+ "/PseudoExemplar"
        self.data_branch = '/10_10'
        self.dataset_name = '/Cifar100_split'
        self.code_root = code_root        
        
        if os.path.exists(self.record_root) ==False:
            os.mkdir(self.record_root)
        
        self.base_classes = 10 
        self.incre_classes = 10
        self.total_classes = 100   
        
        self.feature_dim = 2048

        self.save_cnn_name = "cnn_model"
        # self.save_ss_name = "ss_set"
        # self.save_cvae_name = "cvae_model"
        self.base_cnn_train_his_name = "base_cnn_train"
        # self.base_cvae_train_his_name = "base_cvae_train"

        self.icl_cnn_train_his_name = "icl_cnn_train"
       
        # self.icl_cvae_train_his_name = "icl_cvae_train"        
        # for SS Auto-Encoder
        # self.insert_check_dict = {"en":[1,1,1,1,1],
                                  # "de":[0,1,1,1,1]}
        # self.ss_size_dict = {"en":[2048, 1024, 1024, 512, 512],
                             # "de":[0, 512, 512, 1024, 1024]}
                             
        self.insert_check_list = [1,1,1]
        self.ss_sizes = [2048, 1024, 1024]
        
        self.base_epoch = 15
        self.base_lr = 0.001
        
        
        self.icl_cnn_epoch = 100
        self.icl_cnn_lr_dict = {"base":5e-3, "ss":5e-4}  
        self.icl_cnn_train_his_name = "icl_cnn_train"
        
        # self.icl_loss_ratio = [0.1, 0.09, 0.08, 0.077, 0.071, 0.066, 0.0625, 0.058, 0.055, 0.05] # 50/5
        self.icl_loss_ratio = [ 0.5, 0.5, 0.33, 0.25, 0.2, 0.16, 0.14, 0.125, 0.111] # 10/10
        # self.icl_loss_ratio = [ 0.4, 0.22, 0.15, 0.1, 0.1, 0.09, 0.075, 0.061, 0.05] # 10/10
        
        # ce_w = [0.25, 0.15, 0.15, 0.1, 0.08, 0.06, 0.05, 0.05, 0.04, 0.03]
        # ce_w = [0.1, 0.09, 0.08, 0.077, 0.071, 0.066, 0.0625, 0.058, 0.055, 0.05]
        # ce_w = [w * 4 for w in ce_w ]
        
        # assumption : ratio down while # current classes / # acculmate classes down
        # ce_w = [0.1, 0.09, 0.08, 0.077, 0.071, 0.066, 0.0625, 0.058, 0.055, 0.05]    # for use KD only
        # ce_w = [ 0.5, 0.5, 0.33, 0.25, 0.2, 0.16, 0.14, 0.125, 0.111]
        # ce_w = [w * 2 for w in ce_w ] # for use KL only
        # ce_w = [ 0.7, 0.7, 0.66, 0.5, 0.4, 0.32, 0.28, 0.25, 0.222]
            
        # ce_w = [0.3, 0.25, 0.2, 0.15, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1] # for fixKD 
        # ce_w = [ 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.3, 0.2, 0.2, 0.2]
        # ce_w = [ 0.6, 0.55, 0.5, 0.45, 0.4, 0.35, 0.35, 0.25, 0.25, 0.25] # for fixKL
        # ce_w = [w * 5 for w in ce_w ]# for fixKL
                
        
        self.batch_size = 128
    