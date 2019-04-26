#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  7 14:23:14 2019

@author: usoysal
GET THE FILES

"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan  2 12:53:04 2019

@author: usoysal
"""
#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 31 17:29:33 2018

@author: usoysal
"""

import torch 
import torch.nn as nn
import torchvision.datasets as dsets
from skimage import transform
import torchvision.transforms as transforms
from torch.autograd import Variable
import pandas as pd;
import numpy as np;
from torch.utils.data import Dataset, DataLoader
#from vis_utils import *
import random;
import math;
import matplotlib.pyplot as plt
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sn

num_epochs = 150;
batch_size = 10;
learning_rate = 0.00001;


# This is to be updated
numChannels=20


class MrdsDataset(Dataset):
    '''  '''
    def __init__(self, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the csv file
            transform (callable): Optional transform to apply to sample
        """

        data = pd.read_csv(csv_file);
        self.X = np.array(data.iloc[:, 1])
        self.Y = np.array(data.iloc[:, 0]);

        del data;
        self.transform = transform;

    def __len__(self):
        return len(self.X);

    def __getitem__(self, idx):
        item = self.X[idx];
        label = self.Y[idx];

        if self.transform:
            item = self.transform(item);

        return (item,label)
     
    
mrds_dataset = MrdsDataset(csv_file='mrds_train_ok.csv')

mrds_loader = torch.utils.data.DataLoader(dataset=mrds_dataset,
                                           batch_size=batch_size,
                                           shuffle=True);  
  
                                          
test_dataset = MrdsDataset(csv_file='mrds_test_ok.csv')

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                           batch_size=batch_size,
                                           shuffle=True); 
                                          
                                          
                                          
                                         
data_transform = transforms.Compose([
        transforms.RandomSizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])
#AeroMag_dataset = datasets.ImageFolder(root='data/AeroMag/64x64/',
#                                           transform=data_transform)
#dataset_loader = torch.utils.data.DataLoader(AeroMag_dataset,
#                                             batch_size=4, shuffle=True,
#                                             num_workers=4) 
                                          
class CNN(nn.Module):

    def __init__(self):
        super(CNN, self).__init__()
        # 1 input image channel, 6 output channels, 5x5 square convolution
        # kernel
        self.conv1 = nn.Conv2d(numChannels, 12, 5)
        self.conv2 = nn.Conv2d(12, 16, 5,padding=(1,1))
        self.conv3 = nn.Conv2d(16, 32, 5)
        # an affine operation: y = Wx + b
        self.fc1 = nn.Linear(32 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        # Max pooling over a (2, 2) window
        #64x64 by 3 channel
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2)) # 64x64
        # If the size is a square you can only specify a single number
        #30x30 by 6 channel
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2)) #6@60x60
        #14x14 by 16 channel
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        #5x5 by 32 channel 
        x = x.view(-1, 32 * 5 * 5) # This numbers are subject to change        
        x = F.relu(self.fc1(x))        
        x = F.relu(self.fc2(x))    
        x = self.fc3(x)
  
        return x
    
    def num_flat_features(self, x):
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features

cnn = CNN();                                 
print(cnn)

#loss function and optimizer
criterion = nn.CrossEntropyLoss();
optimizer = torch.optim.Adam(cnn.parameters(), lr=learning_rate);
losses = [];

cube=[]
for epoch in range(num_epochs):
    for i, (item, label) in enumerate(mrds_loader):
        #images = Variable(images.float())
        #labels = Variable(labels.long())      
        # Forward + Backward + Optimize
        batch_len=len(item)
        #print(item)
        #print(label)
        cube_batch=np.zeros([batch_len,numChannels,64,64])
        batch_counter=0
        for j in item:
            #print(i.type)
            j_string=str(j.item())
            #print(i_string)
            aa1=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/AeroMag/AeroMag_'+str(j_string)+'.tif')
            aa2=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/GravityAnomalyBouguer/GravityAnomalyBouguer_'+str(j_string)+'.tif')
            aa3=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/GravityAnomalyIsostatic/GravityAnomalyIsostatic_'+str(j_string)+'.tif')
            aa4=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/DEM/DEM_'+str(j_string)+'.tif')
            aa5=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/URadiometric/URadiometric_'+str(j_string)+'.tif')
            aa6=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/ThRadiometric/ThRadiometric_'+str(j_string)+'.tif') 
            aa7=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/KRadiometric/KRadiometric_'+str(j_string)+'.tif')







        
    
            aa1_numpy = np.array(aa1)
            aa2_numpy = np.array(aa2)
            aa3_numpy = np.array(aa3)
            aa4_numpy = np.array(aa4)
            aa5_numpy = np.array(aa5)
            aa6_numpy = np.array(aa6)
            aa7_numpy = np.array(aa7)
            
            cube=np.dstack((aa1_numpy,aa2_numpy,aa3_numpy,aa4_numpy,aa5_numpy,aa6_numpy,aa7_numpy))
            #THIS LINE IS WRONG SHOULD BE CHANGED
            cube=np.reshape(cube,(numChannels,64,64))
            cube_batch[batch_counter,:,:,:]=cube
            batch_counter=batch_counter+1
        cube_batch_tensor=torch.from_numpy(cube_batch).float() # Tensor
        #print(i)
        optimizer.zero_grad()
        outputs = cnn(cube_batch_tensor)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()
        
        losses.append(loss.data);
        
        if (i+1) % 100 == 0:
            print ('Epoch : %d/%d, Iter : %d/%d,  Loss: %.4f' 
                   %(epoch+1, num_epochs, i+1, len(mrds_dataset)//batch_size, loss.data))


torch.save(cnn,"/home/ubuntu/kynesfield/scripts/py/umut/model.pt")
            
# PREDICTION TRAIN
correct = 0
total = 0   
for i, (item, label) in enumerate(mrds_loader):
    batch_len=len(item)
    #print(item)
    #print(label)
    cube_batch=np.zeros([batch_len,numChannels,64,64])
    batch_counter=0
    for j in item:
        #print(i.type)
        j_string=str(j.item())
        #print(i_string)
        a1=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/AeroMag/AeroMag_'+str(j_string)+'.tif')
        aa2=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/GravityAnomalyBouguer/GravityAnomalyBouguer_'+str(j_string)+'.tif')
        aa3=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/GravityAnomalyIsostatic/GravityAnomalyIsostatic_'+str(j_string)+'.tif')
        aa4=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/DEM/DEM_'+str(j_string)+'.tif')
        aa5=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/URadiometric/URadiometric_'+str(j_string)+'.tif')
        aa6=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/ThRadiometric/ThRadiometric_'+str(j_string)+'.tif') 
        aa7=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/KRadiometric/KRadiometric_'+str(j_string)+'.tif')
        
    
        aa1_numpy = np.array(aa1)
        aa2_numpy = np.array(aa2)
        aa3_numpy = np.array(aa3)
        aa4_numpy = np.array(aa4)
        aa5_numpy = np.array(aa5)
        aa6_numpy = np.array(aa6)
        aa7_numpy = np.array(aa7)
            
        cube=np.dstack((aa1_numpy,aa2_numpy,aa3_numpy,aa4_numpy,aa5_numpy,aa6_numpy,aa7_numpy))   
        #THIS LINE IS WRONG SHOULD BE CHANGED
        cube=np.reshape(cube,(numChannels,64,64))
        cube_batch[batch_counter,:,:,:]=cube
        
        batch_counter=batch_counter+1
    cube_batch_tensor=torch.from_numpy(cube_batch).float() # Tensor
        #print(i)
    outputs = cnn(cube_batch_tensor)
    _, predicted = torch.max(outputs.data, 1)
    total += label.size(0)
    correct += (predicted == label).sum().item()
    print('Accuracy of the network on the train images: %d %%' % (
                    100 * correct / total))
            
        
        
        
        
# PREDICTION TEST
    
correct = 0
total = 0 
true_success=0
true_failure=0
prediction=[]
true_label=[]
    
 
for i, (item, label) in enumerate(test_loader):
    batch_len=len(item)
    #print(item)
    #print(label)
    cube_batch=np.zeros([batch_len,numChannels,64,64])
    batch_counter=0
    for j in item:
        #print(i.type)
        j_string=str(j.item())
        #print(i_string)
        a1=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/AeroMag/AeroMag_'+str(j_string)+'.tif')
        aa2=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/GravityAnomalyBouguer/GravityAnomalyBouguer_'+str(j_string)+'.tif')
        aa3=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/GravityAnomalyIsostatic/GravityAnomalyIsostatic_'+str(j_string)+'.tif')
        aa4=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/DEM/DEM_'+str(j_string)+'.tif')
        aa5=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/URadiometric/URadiometric_'+str(j_string)+'.tif')
        aa6=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/ThRadiometric/ThRadiometric_'+str(j_string)+'.tif') 
        aa7=Image.open('/home/ubuntu/kynesfield/datasets/clipped/64x64/KRadiometric/KRadiometric_'+str(j_string)+'.tif')
        
    
        aa1_numpy = np.array(aa1)
        aa2_numpy = np.array(aa2)
        aa3_numpy = np.array(aa3)
        aa4_numpy = np.array(aa4)
        aa5_numpy = np.array(aa5)
        aa6_numpy = np.array(aa6)
        aa7_numpy = np.array(aa7)
            
        cube=np.dstack((aa1_numpy,aa2_numpy,aa3_numpy,aa4_numpy,aa5_numpy,aa6_numpy,aa7_numpy))   
        #THIS LINE IS WRONG SHOULD BE CHANGED
        cube=np.reshape(cube,(numChannels,64,64))
        cube_batch[batch_counter,:,:,:]=cube
        
        batch_counter=batch_counter+1
    cube_batch_tensor=torch.from_numpy(cube_batch).float() # Tensor
        #print(i)
    outputs = cnn(cube_batch_tensor)
    _, predicted = torch.max(outputs.data, 1)
    predicted_numpy=predicted.numpy()
    true_label_numpy=label.numpy()
    prediction=np.concatenate((prediction, predicted_numpy))
    true_label=np.concatenate((true_label, true_label_numpy))
    total += label.size(0)
    correct += (predicted == label).sum().item()
    print('Accuracy of the network on the test images: %d %%' % (
                    100 * correct / total))
        

        
y_true = true_label
y_pred = prediction
cnf_matrix=confusion_matrix(y_true, y_pred)


df_cm = pd.DataFrame(cnf_matrix, index = [i for i in "FT"],
                  columns = [i for i in "FT"])
plt.figure(figsize = (10,7))
sn.heatmap(df_cm, annot=True)

print(cnf_matrix)
   
np.savetxt("test_labels.csv", y_true, delimiter=",")   
np.savetxt("test_prediction.csv", y_pred, delimiter=",") 
np.savetxt("conf.csv", cnf_matrix, delimiter=",")   
          
        
        
        
