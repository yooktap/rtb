import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split

#Rename columns
column_names = {0:'Filename'}
for i in range(1, 133):
  column_names[((i-1)*4)+1]=(f'Part{i}_vol')
  column_names[((i-1)*4)+2]=(f'Part{i}_csf')
  column_names[((i-1)*4)+3]=(f'Part{i}_gm')
  column_names[((i-1)*4)+4]=(f'Part{i}_wm')
column_names[529]='out_class'

df = pd.read_csv('data_points.csv', header=None)
df.rename(columns=column_names, inplace=True)


pd.set_option("display.max_columns", None)
pd.set_option("display.max_rows", None)

#drop invalid data
drop_list = [2, 56, 68, 86, 96]
df.drop(drop_list, inplace=True)

csf_columns = [x for x in list(df.columns.values) if "csf" in x]
df.drop(csf_columns, axis=1, inplace=True)

for i in range(1, 133):
  df[f'Part{i}_gm'] = df[f'Part{i}_gm'] / df[f'Part{i}_vol']
  df[f'Part{i}_wm'] = df[f'Part{i}_wm'] / df[f'Part{i}_vol']

vol_columns = [x for x in list(df.columns.values) if "vol" in x]
df.drop(vol_columns, axis=1, inplace=True)

#Drop SpD data
df = df[df['out_class'] != 2]

#get ready for train test split
#convert dataframe to numpy array
X = df.drop(['Filename', 'out_class'], axis=1).values
y = df['out_class'].values

#train test split
torch.manual_seed(32)
Xorig_train, Xorig_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=32)

#create model class
class Model(nn.Module):
  def __init__(self, in_features=2, h1=8, h2=9, out_features=2):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)
    return x

parti_train_acc = {}
parti_test_acc = {}
#train and test model for each part separately
for parti in range (1, 133):
    print(f"Training and Testing...Part{parti}")
    
    #get features
    X_train = Xorig_train[:,((2*parti)-2):(2*parti)]
    X_test = Xorig_test[:,((2*parti)-2):(2*parti)]

    #convert X features to float tensors
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    #convert y features to long tensors
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)


    #instantiate model
    in_feat=X_train.shape[1]
    model = Model(in_features=in_feat)

    #set criterion to measure loss
    criterion = nn.CrossEntropyLoss()
    #choose adamax optimizer (good with noisy data)
    optimizer = torch.optim.Adamax(model.parameters(), lr=0.001)

    #train model
    epochs = 100000
    losses = []
    for i in range(epochs):
        y_pred = model.forward(X_train)
        loss = criterion(y_pred, y_train)
        #print(loss)
        losses.append(loss.detach().numpy())

        #print losses
        if i % 10000 == 0:
            print(f'Epoch: {i} Loss: {loss}')

        #back propagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    #validate with test dataset
    print(f"\n{parti} Train")
    with torch.no_grad():
        y_eval = model.forward(X_train)
        loss = criterion(y_eval, y_train)
        print(loss)
        correct=0
        for i in range(len(y_eval)):
            #print(torch.argmax(y_eval[i]), y_train[i], torch.argmax(y_eval[i]) == y_train[i])
            if torch.argmax(y_eval[i]) == y_train[i]:
                correct += 1
        print(correct)
        print(y_train.shape[0])
        print(correct/y_train.shape[0])
        parti_train_acc[parti] = correct/y_train.shape[0]


    print(f"\n{parti} Test")
    with torch.no_grad():
        y_eval = model.forward(X_test)
        loss = criterion(y_eval, y_test)
        print(loss)
        correct=0
        for i in range(len(y_eval)):
            #print(torch.argmax(y_eval[i]), y_test[i], torch.argmax(y_eval[i]) == y_test[i])
            if torch.argmax(y_eval[i]) == y_test[i]:
                correct += 1
        print(correct)
        print(y_test.shape[0])
        print(correct/y_test.shape[0])
        parti_test_acc[parti] = correct/y_test.shape[0]
    
print(parti_train_acc)
sorted_test_acc = dict(sorted(parti_test_acc.items(), key=lambda item: item[1]))
for parti in sorted_test_acc:
    print(f"Part {parti} Test Accuracy: {parti_test_acc[parti]}, Train Accuracy: {parti_train_acc[parti]}")

    #save model
    #torch.save(model.state_dict(), 'model_name.pt')
    #load model
    #load_model = Model()
    #load_model.load_state_dict(torch.load('model_name.pt'))