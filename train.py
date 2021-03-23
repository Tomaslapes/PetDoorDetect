import torch
import torchvision.transforms as transforms
import torch.optim as optim
import torch.nn as nn
from loadData import MyDataset
from torch.utils.data import DataLoader
from model import Model
import seaborn as sea
from tqdm import tqdm
import pandas as pd
import matplotlib.pyplot as plt


# DEFINE HYPER PARAMETERS
lr = 0.00018
EPOCHS = 3
batch_size = 24


# SET DEVICE
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# CREATE A MODEL
model = Model()
model.to(device)

# LOAD DATA
train_dataset = MyDataset({0:["Nothing"],1:["Cat","Dog","Both"]},transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize((100,100))
]))
test_dataset = MyDataset({0:["Empty"],1:["Present"]},path="TestData/",transform=transforms.Compose([
    transforms.ToTensor(),
    transforms.Grayscale(),
    transforms.Resize((100,100))
]))
train_loader = DataLoader(dataset=train_dataset,shuffle = True,batch_size = batch_size)
test_loader = DataLoader(dataset=test_dataset,shuffle = True,batch_size = 1)

# TRAIN THE NETWORK
optimizer = optim.Adam(model.parameters(),lr=lr)
lossFunction = nn.BCEWithLogitsLoss() # Binary cross entropy loss
loss_log = []

for epoch in range(EPOCHS):
    print(f"\n[EPOCH]: {epoch+1}")
    for data,labels in tqdm(train_loader):
        data = data.to(device)
        labels = labels.to(device)

        # Forward
        preds = model(data.float())
        loss = lossFunction(preds,labels.float())
        loss_log.append(loss.item())
        print(loss)

        # Backward
        optimizer.zero_grad()
        loss.backward()

        # Gradient descent / Adam
        optimizer.step()

# Validation
print("[VALIDATION] Starting validation")

model.eval()

total_correct = 0
total = 0
for test_data,test_label in tqdm(test_loader):
    test_data = test_data.to(device)
    test_label = test_label.to(device)

    # forward
    preds = model(test_data.float())
    x = (preds >= 0.5) == test_label
    if x:
        total_correct +=1
    total +=1


print(f"{total_correct} out of {total} or [{(total_correct/total)*100}%]")


print(loss_log)

# Graph the loss over time/iterations
sea.set_theme()
plot_data = pd.DataFrame({"loss":loss_log,'x': [i for i in range(len(loss_log))]})
sea.relplot(data=plot_data,x='x',y='loss',kind='line')
plt.show()


# Save the model
should_save = input("Save? Y/n")
name = input("Please enter a name")


if should_save == 'Y':
    torch.save(model.to(device),f"{name}.pth")
    if input("Save also state dictionary? Y/n")=="Y":
        save_obj = model.state_dict()
        torch.save(save_obj, f"{name+'_state_dict'}.pth")


