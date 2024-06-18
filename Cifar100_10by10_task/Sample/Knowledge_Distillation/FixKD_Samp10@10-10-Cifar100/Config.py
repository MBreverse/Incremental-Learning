#config
import os

code_root = "./Distill_with_OldData_10by10_10"
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
        
        self.base_epoch = 30
        self.base_lr = 0.001
        
        
        self.icl_cnn_epoch = 150
        self.icl_cnn_lr_dict = {"base":1e-4, "ss":5e-4}  
        self.icl_cnn_train_his_name = "icl_cnn_train"
        
        self.icl_loss_ratio = [0.1, 0.09, 0.08, 0.077, 0.071, 0.066, 0.0625, 0.058, 0.055, 0.05]

        self.batch_size = 1024
    