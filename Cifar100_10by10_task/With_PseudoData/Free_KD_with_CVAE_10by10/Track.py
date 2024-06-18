#Tracker of training
class LossTracker():
    def __init__(self, num_loss = int(0), loss_names = []):
        
        self.loss_names = loss_names
        
        self.loss_his_dict = {}
        for name in loss_names:
            self.loss_his_dict.update({name : [] })
        
        self.loss_accu_dict = {}
        for name in loss_names:
            self.loss_accu_dict.update({name : 0})
            
        self.batch_count = 0
            
    def update(self):
        
        for i, name in enumerate( self.loss_names ):
            self.loss_his_dict[name].append(self.loss_accu_dict[name]/ self.batch_count)
        
        self.zerolize()
    
    def accumlate(self, tuple_of_loss = ()):
        
        self.batch_count += 1
        
        for i, name in enumerate( self.loss_names ):
            self.loss_accu_dict[name] += tuple_of_loss[i]    

    def get_loss_history(self, name):
        
        return self.loss_his_dict[name]

    def get_epoch_loss(self, name):
        
        return self.loss_accu_dict[name]/ self.batch_count    

    def zerolize(self):
        
        self.batch_count = 0
        for name in  self.loss_names:
            self.loss_accu_dict[name] = 0    

class AccTracker():
    def __init__(self):
        
        self.num_correct_sample = 0
        self.num_total_sample = 0 
        self.acc_history = [] 
        
    def accumlate(self, cor_sample, num_sample):
        
        self.num_correct_sample += cor_sample
        self.num_total_sample += num_sample
        
    def update(self):
        self.acc_history.append(self.num_correct_sample / self.num_total_sample) 
        self.zerolize()        
    
    def zerolize(self):    
        self.num_correct_sample = 0
        self.num_total_sample = 0  
        
    def get_epoch_acc(self):
        return self.num_correct_sample / self.num_total_sample
    
    def get_acc_history(self):
        return self.acc_history
        