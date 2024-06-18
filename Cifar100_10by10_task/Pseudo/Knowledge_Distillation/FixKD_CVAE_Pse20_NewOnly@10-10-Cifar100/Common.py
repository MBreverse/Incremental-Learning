#Common file

# Generate Pseudo Feature by VAE with variance and mean
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


            torch.set_printoptions(threshold=np.inf)
        
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

# Generate Pseudo Feature by CVAE with variance and mean
class CVAE_PseudoExempar(Dataset):
    
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

            instance = std * torch.randn([samples, std.shape[0]]) + mean

            instance = instance.to(device).float()

            #decode prototype to the feature size 
            labels = torch.zeros( int(sample_per_class) ) + c
            labels = labels.to(torch.int64)
            onehot = one_hot(labels, 10)
            instance = AE.decode(instance, onehot)
            instance = instance.detach()
            instance = instance.view(-1,784)

            all_data_list.append(instance.cpu())             
            all_label_list.append(labels.cpu())

            torch.set_printoptions(threshold=np.inf)
        
        self.all_data = torch.cat(all_data_list, dim = 0)
        self.all_labels = torch.cat(all_label_list, dim = 0)
        
        all_data_list.clear()
        all_label_list.clear()

    
    def __getitem__(self, index):
    
        prototype = self.all_data[index]
        target = self.all_labels[index]
        

        return prototype.view(-1,28,28), target

    def __len__(self):
    
        return(self.all_labels.size()[0])  

# Generate Pseudo Feature by CVAE without variance and mean
class CVAE_PseudoExempar(Dataset):
    
    def __init__(self, sample_per_class, class_digits, task):
        super(CVAE_PseudoExempar, self).__init__()
        '''
        class_digits = list [0,49] 
        '''
        # all_mean = torch.load('./PseudoExemplar/MeanCov/task{}_feature_mean.pth.tar'.format(task))
        # all_std = torch.load('./PseudoExemplar/MeanCov/task{}_feature_cov.pth.tar'.format(task))
        
        record_root = './record'        
        
        model = CVAE(28*28, cfg.z_dim, 10).to(device)  
        
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



