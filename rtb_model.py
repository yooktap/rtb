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


#create model class
class Model(nn.Module):
  def __init__(self, in_features=6, h1=8, h2=9, out_features=2):
    super().__init__()
    self.fc1 = nn.Linear(in_features, h1)
    self.fc2 = nn.Linear(h1, h2)
    self.out = nn.Linear(h2, out_features)

  def forward(self, x):
    x = F.relu(self.fc1(x))
    x = F.relu(self.fc2(x))
    x = self.out(x)
    return x


interesting_columns = [x for x in list(df.columns.values) if (x in ["Part49_gm", "Part49_wm", "Part106_gm", "Part106_wm", "Part130_gm", "Part130_wm", "out_class"])]
df = df[interesting_columns]

#get ready for train test split
#convert dataframe to numpy array
X = df.drop(['out_class'], axis=1).values
y = df['out_class'].values


pd.set_option('display.width', 1000)
print(df.describe())

#train test split
rseed=32
torch.manual_seed(rseed)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=rseed)

#convert X features to float tensors
X_train = torch.FloatTensor(X_train)
X_test = torch.FloatTensor(X_test)
#convert y features to long tensors
y_train = torch.LongTensor(y_train)
y_test = torch.LongTensor(y_test)


#instantiate model
in_feat=X_train.shape[1]
print(f"Input features: {in_feat}")
model = Model(in_features=in_feat)

#set criterion to measure loss
criterion = nn.CrossEntropyLoss()
#choose adamax optimizer (good with noisy data)
optimizer = torch.optim.Adamax(model.parameters(), lr=0.00065)

#train model
epochs = 90000
losses = []
for i in range(epochs):
    y_pred = model.forward(X_train)
    loss = criterion(y_pred, y_train)
    #print(loss)
    losses.append(loss.detach().numpy())

    

    #back propagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    #print losses
    if (i % 5000 == 0) and (i != 0):
        print(f'Epoch: {i} Loss: {loss}')
        with torch.no_grad():
            y_eval = model.forward(X_train)
            loss = criterion(y_eval, y_train)
            print(loss)
            correct=0
            for i in range(len(y_eval)):
                #print(y_train[i], y_eval[i], torch.argmax(y_eval[i]), torch.argmax(y_eval[i]) == y_train[i])
                if torch.argmax(y_eval[i]) == y_train[i]:
                    correct += 1
            print(f"Train Correct: {correct} Total: {y_train.shape[0]} Train Accuracy: {correct/y_train.shape[0]}")
            

            y_eval = model.forward(X_test)
            loss = criterion(y_eval, y_test)
            print(loss)
            correct=0
            for i in range(len(y_eval)):
                print(y_test[i], y_eval[i], torch.argmax(y_eval[i]), torch.argmax(y_eval[i]) == y_test[i])
                if torch.argmax(y_eval[i]) == y_test[i]:
                    correct += 1
            print(f"Test Correct: {correct} Total: {y_test.shape[0]} Test Accuracy: {correct/y_test.shape[0]}\n")

#save model
torch.save(model.state_dict(), 'model_49_76_106_best.pt')
    #load model
    #load_model = Model()
    #load_model.load_state_dict(torch.load('model_name.pt'))
