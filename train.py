###Custom implementation of an RNN with identity mapping from hidden to output for reduced order modelling

from torch import nn
import torch
import numpy as np
from tqdm import trange


class ParConcatenator(nn.Module):

    def __init__(self):
        super(ParConcatenator,self).__init__()

    def forward(self,x,y):
        return torch.concat((x,y.unsqueeze(1).repeat(1,x.shape[1],1)),axis=2)

class PredictFirst(nn.Module):
    def __init__(self, par_size, pde_size, windows_size=1 ,hidden_mult=2,dropout=0.05):
        super(PredictFirst,self).__init__()
        self.windows_size=1
        self.hidden_dim=100#windows_size*pde_size
        self.nn_first=nn.Sequential(nn.Linear(par_size,self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, pde_size))


    def forward(self, theta):
        return self.nn_first(theta)

class SequetialPrediction(nn.Module):
    def __init__(self, par_size, pde_size, windows_size=1 ,hidden_mult=2,dropout=0.05):
        super(SequetialPrediction,self).__init__()
        self.windows_size=windows_size
        self.hidden_dim=100#windows_size*pde_size
        self.concatenator=ParConcatenator()
        self.nn_first=nn.Sequential(nn.Linear(self.windows_size*pde_size+par_size,self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, self.hidden_dim),nn.ReLU(),nn.Dropout(dropout),
                                    nn.Linear(self.hidden_dim, pde_size))


    def forward(self, theta, x):
        x=self.concatenator(x,theta)
        x=x.reshape(-1,x.shape[2])
        return self.nn_first(x)


class TimeDependentROM(nn.Module):
    def __init__(self,par_size, pde_size, windows_size=1 ,hidden_mult=2,dropout=0.05):
        super(TimeDependentROM,self).__init__()
        self.firstnn=PredictFirst(par_size,pde_size,windows_size,hidden_mult,dropout)
        self.seqnn=SequetialPrediction(par_size,pde_size,windows_size,hidden_mult,dropout)

    def predict_first(self,theta):
        return self.firstnn(theta)

    def predict_next(self,theta,x):
        return self.seqnn(theta,x)
    
    def predict_grop(self,theta,steps):
        x=self.predict_first(theta)
        all=torch.zeros_like(x.unsqueeze(1).repeat(1,steps,1))
        all[:,0,:]=x
        for i in range(1,steps):
            x=self.predict_next(theta,x.unsqueeze(1))
            all[:,i,:]=x
        return all



def process_dataset(windows_size,solutions):
    print(solutions.shape)
    start=solutions[:,:windows_size,:]
    inputs=torch.zeros(start.shape[0],solutions.shape[1]-windows_size,windows_size*solutions.shape[2])
    outputs=torch.zeros(start.shape[0],solutions.shape[1]-windows_size,windows_size*solutions.shape[2])
    for i in range(solutions.shape[1]-windows_size):
        inputs[:,i]=solutions[:,i:i+windows_size,:].reshape(start.shape[0],-1)
        outputs[:,i]=solutions[:,i+windows_size,:].reshape(start.shape[0],-1)

    return start,inputs,outputs



BATCH_SIZE=20
TRAIN_SIZE=80
TEST_SIZE=20
WINDOWS_SIZE=1

outputs=torch.tensor(np.load("outputs.npy")).float()
outputs_bak=outputs.clone()
parameters=torch.tensor(np.load("inputs.npy")).float()[:,0,1].reshape(-1,1)

start,inputs,outputs=process_dataset(1,outputs)

print(start.shape)
print(torch.linalg.norm(start-torch.mean(start,axis=0))/torch.linalg.norm(start))


parameters_train=parameters[:TRAIN_SIZE]
parameters_test=parameters[TRAIN_SIZE:]
start_train=start[:TRAIN_SIZE]
start_test=start[TRAIN_SIZE:]

outputs_train=outputs[:TRAIN_SIZE]
outputs_test=outputs[TRAIN_SIZE:]
outputs_pred_train=outputs_train[:,-1,:]
outputs_train=outputs_train[:,:-1,:]
outputs_pred_test=outputs_test[:,-1,:]
outputs_test=outputs_test[:,:-1,:]


inputs_train=inputs[:TRAIN_SIZE]
inputs_pred_train=inputs_train[:,-1,:]
inputs_train=inputs_train[:,:-1,:]

inputs_test=inputs[TRAIN_SIZE:]
inputs_pred_test=inputs_test[:,-1,:].unsqueeze(1)
inputs_test=inputs_test[:,:-1,:]


dataset_train=torch.utils.data.TensorDataset(start_train,inputs_train,outputs_train,parameters_train)
dataloader_train=torch.utils.data.DataLoader(dataset_train,batch_size=BATCH_SIZE)
dataset_test=torch.utils.data.TensorDataset(start_test,inputs_test,outputs_test,parameters_test,outputs_pred_test,inputs_pred_test)
dataloader_test=torch.utils.data.DataLoader(dataset_test,batch_size=BATCH_SIZE)


model = TimeDependentROM(par_size=1,windows_size=1,pde_size=outputs_train.shape[2])
n_epochs = 3000
lr=0.0001

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.firstnn.parameters(), lr=lr)

for epoch in trange(0, n_epochs):
    for data in dataloader_train:
        start,inputs,outputs,par=data
        optimizer.zero_grad()
        start_hat = model.predict_first(par).reshape(start.shape)
        loss = criterion(start_hat,start)
        loss.backward() 
        optimizer.step()
    print(torch.linalg.norm(start_hat.reshape(BATCH_SIZE,-1)-start.reshape(BATCH_SIZE,-1))/torch.linalg.norm(start.reshape(BATCH_SIZE,-1)))
    for data in dataloader_test:
        start,inputs,outputs,par,out_pred,inputs_pred=data
        start_hat = model.predict_first(par).reshape(start.shape)
    print(torch.linalg.norm(start_hat.reshape(BATCH_SIZE,-1)-start.reshape(BATCH_SIZE,-1))/torch.linalg.norm(start.reshape(BATCH_SIZE,-1)))




counter=0
loss_test=0
loss_test_start=0
pred_error=0
model.firstnn.eval()
for data in dataloader_test:
    start,inputs,outputs,par,out_pred,inputs_pred=data
    start_hat = model.predict_first(par).reshape(start.shape)
    print("First step relative error is", torch.linalg.norm(start_hat.reshape(BATCH_SIZE,-1)-start.reshape(BATCH_SIZE,-1))/torch.linalg.norm(start.reshape(BATCH_SIZE,-1)))


n_epochs = 3000
lr=0.0001

criterion = nn.MSELoss()
optimizer = torch.optim.AdamW(model.seqnn.parameters(), lr=lr)

for epoch in trange(0, n_epochs):
    for data in dataloader_train:
        start,inputs,outputs,par=data
        outputs=outputs.reshape(-1,outputs.shape[2])
        optimizer.zero_grad()
        outputs_hat = model.predict_next(par,inputs).reshape(outputs.shape)
        loss = criterion(outputs_hat,outputs)
        loss.backward() 
        optimizer.step()
    print(torch.linalg.norm(outputs_hat.reshape(BATCH_SIZE,-1)-outputs.reshape(BATCH_SIZE,-1))/torch.linalg.norm(outputs.reshape(BATCH_SIZE,-1)))
    for data in dataloader_test:
        start,inputs,outputs,par,outputs_pred,inputs_pred=data
        outputs_hat = model.predict_next(par,inputs).reshape(outputs.shape)
    print(torch.linalg.norm(outputs_hat.reshape(BATCH_SIZE,-1)-outputs.reshape(BATCH_SIZE,-1))/torch.linalg.norm(outputs.reshape(BATCH_SIZE,-1)))





counter=0
loss_test=0
loss_test_start=0
pred_error=0
model.seqnn.eval()
for data in dataloader_test:
    start,inputs,outputs,par,outputs_pred,inputs_pred=data
    outputs_hat = model.predict_next(par,inputs).reshape(outputs.shape)
    print("Known next rel error is ", torch.linalg.norm(outputs_hat.reshape(BATCH_SIZE,-1)-outputs.reshape(BATCH_SIZE,-1))/torch.linalg.norm(outputs.reshape(BATCH_SIZE,-1)))

outputs_pred_hat=model.predict_next(par,inputs_pred)
print("Unkown next rel error is ",torch.linalg.norm(outputs_pred_hat.reshape(BATCH_SIZE,-1)-outputs_pred.reshape(BATCH_SIZE,-1))/torch.linalg.norm(outputs_pred.reshape(BATCH_SIZE,-1)))

all_pred=model.predict_grop(parameters,102)
print(all_pred.shape)
print(outputs_bak.shape)
print(torch.linalg.norm(outputs_bak-all_pred)/torch.linalg.norm(outputs_bak))
