import torch
from ModelCollection import *
from SupportFunction import *
from DatasetProcess import *


a = int(1)
print(a)

# AE = AutoEncoder()
# torch.save(AE.state_dict(), "AE.pth.tar")

# data_root = './data'
# branch = '/50_5by10/feature_set'    
# dataset_path = './data' + '/Cifar100_split'+ branch +'//split'+ "2" +'_dataset.pth.tar'    

# dataset = torch.load(dataset_path)


# dset = Shift_Label_Dataset(dataset["train_true"], origin_class_logit_range = [50,54], target_class_logit_range = [1,5])
# print(dset.stack_label)

# x = dataset["stack_feature]
# print(dataset["train_true"])
# exemplar_set = Exemplar_Feature_Set(dataset)
# # torch.save()
# torch.save(exemplar_set,'exemplar_dataset.pth.tar')
# # Write_Single_Task_TestAcc_Record(root = './record/', task_acc_list =[0.1,0.2,0.3,0.4], train_task = 4)

# cnn = CNN()
# # x = torch.zeros(1,2048)
# print(list(cnn.children()))
# for i, module in enumerate(list[cnn.children()][1:]):
    # print(module)
    

    

# insertion_check_list = [1,0,1,1,1]
# sizes_of_kernel_set = [2048,1024,1024,1024,1024]
# mks1 = Multi_Scale_Shift_Kernel(insertion_check_list, sizes_of_kernel_set)
# mks2 = Multi_Scale_Shift_Kernel(insertion_check_list, sizes_of_kernel_set)
# model = SS_Model(cnn, insertion_check_list, sizes_of_kernel_set)

# print(len(cnn.state_dict().keys()))
# print(len(mks1.state_dict().keys()))
# print(len(mks2.state_dict().keys()))
# print(len(model.state_dict().keys()))

# count = 0 
# for p in cnn.parameters():
    # count+=1

# print(count)

# model.eval()
# model = model.to("cuda")
# x = torch.ones([1,3,224,224]).to("cuda")
# y = model(x)
# print(type(y))
# print(y)



# d1 = {"a": 1}
# print(d1.keys())
# for key in d1.keys():
    # print(key)
    # print(d1[key])
    
# a = torch.ones(10,2048)    
# b = torch.ones(3,2048)
# c = a*b 
# print(c.shape)