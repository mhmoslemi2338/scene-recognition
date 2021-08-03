
import os
from keras.preprocessing.image import ImageDataGenerator
from data_augmentation import augment_func
from functions import fit_model , show_result ,fit_model_torch
from networks import network1 , network2 , network3 , network4 , network5
import torch
import torchvision
from importlib.machinery import SourceFileLoader




train_path=os.path.join(os.path.abspath("Data"),"Train")
test_path=os.path.join(os.path.abspath("Data"),"Test")


try:
    f = open(os.path.join(train_path,"done.txt"), "r")
    f.close()
except:
    augment_func()
    

    
train_data =ImageDataGenerator(rescale=1. / 255)
train = train_data.flow_from_directory(train_path, target_size=(225,225), batch_size=64,class_mode='categorical')

test_data =ImageDataGenerator(rescale=1. / 255)
test = test_data.flow_from_directory(test_path, target_size=(225,225), batch_size=64,class_mode='categorical')


############ part 1 ############
net1=network1(train[0][0].shape[1:])
[net1 ,train_hist ,test_hist]=fit_model(epoch=40 , model=net1 , Train=train ,Test=test,path='network1')
show_result(train_hist,test_hist,"network1_plot")


############ part 2 ############
net2=network2(train[0][0].shape[1:])
[net2 ,train_hist ,test_hist]=fit_model(epoch=40 , model=net2 , Train=train ,Test=test,path='network2')
show_result(train_hist,test_hist,"network2_plot")


############ part 3 ############
net3=network3(train[0][0].shape[1:])
[net3 ,train_hist ,test_hist]=fit_model(epoch=45 , model=net3 , Train=train ,Test=test,path='network3')
show_result(train_hist,test_hist,"network3_plot")


########### part 4 ############

 
trans_img = torchvision.transforms.Compose([torchvision.transforms.Resize(224),
                                            torchvision.transforms.CenterCrop(224),
                                            torchvision.transforms.ToTensor()])

train_data = torchvision.datasets.ImageFolder(train_path, transform=trans_img)
train_data = torch.utils.data.DataLoader(train_data, batch_size=64, shuffle=True)

test_data = torchvision.datasets.ImageFolder(test_path, transform=trans_img)
test_data  = torch.utils.data.DataLoader(test_data, batch_size=64, shuffle=True)


print("found "+str(len(train_data.dataset))+" train images in "+str(len(train_data.dataset.classes))+" classes.")
print("found "+str(len(test_data.dataset))+" test images in "+str(len(test_data.dataset.classes))+" classes.")


alex4=network4()
opt = torch.optim.SGD(alex4[20].parameters(), lr=0.003, momentum=0.9)

path_=os.path.abspath("network4")
try:
  os.mkdir(path_)
except:
  pass
[alex4 ,train_hist ,test_hist]=fit_model_torch(epochs=45 , model=alex4 , optimizer=opt , Train=train_data , Test=test_data , path=path_)    
show_result(train_hist,test_hist,"network4_plot")


########### part 5 ############

alex5=network5()
opt = torch.optim.SGD(alex5.parameters(), lr=0.003, momentum=0.9)

path_=os.path.abspath("network5")
try:
  os.mkdir(path_)
except:
  pass
[alex5 ,train_hist ,test_hist]=fit_model_torch(epochs=45 , model=alex5 , optimizer=opt , Train=train_data , Test=test_data , path=path_)    
show_result(train_hist,test_hist,"network5_plot")

