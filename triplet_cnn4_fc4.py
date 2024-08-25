import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

import os
import time
import pickle
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import random

#%% Torch device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))

#%% Data Loading

# pickle_file_path = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\Codes\hack2024\image_array_dict_970368.pkl"
# pickle_file_path = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\Codes\hack2024\image_array_dict_970618_small.pkl"
pickle_file_path = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\Codes\hack2024\image_array_dict_971419_60x60.pkl"

time_start = time.time()

with open(pickle_file_path, 'rb') as f:
    data_dict = pickle.load(f)

print(f"Data read in {time.time() - time_start} seconds.")
#%% Dataset Class

class ImageDataset(Dataset):
    def __init__(self, data_dict, num_data=50000):
        self.data_dict = data_dict
        self.data = dict(sorted(data_dict.items()))
        self.triplets = self._create_triplets()

    def _create_triplets(self):
        triplets = []
        filenames = list(self.data_dict.keys())

        for i in range(0, len(filenames)-2, 3):
            fname_prv, fname_cur, fname_nxt = filenames[i], filenames[i+1], filenames[i+2]

            if fname_prv.split("_")[0] == fname_cur.split("_")[0] == fname_nxt.split("_")[0]:
                if int(fname_prv.split("_")[1][1:]) +1 == int(fname_cur.split("_")[1][1:]) == int(fname_nxt.split("_")[1][1:]) -1:

                    data_prv = torch.from_numpy(self.data[fname_prv] / 255).float().unsqueeze(0)
                    data_cur = torch.from_numpy(self.data[fname_cur] / 255).float().unsqueeze(0)
                    data_nxt = torch.from_numpy(self.data[fname_nxt] / 255).float().unsqueeze(0)

                    triplets.append((
                        fname_prv, fname_cur, fname_nxt, data_prv, data_cur, data_nxt
                    ))
        return triplets



    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

#%% Neural Network

class SiameseTripletNetwork(nn.Module):
    def __init__(self, dropout_rate=0.5):
        super(SiameseTripletNetwork, self).__init__()

        # Convolutional layers
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding='same')
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding='same')
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding='same')
        self.conv4 = nn.Conv2d(128, 256, kernel_size=3, padding='same')

        # Dropout layers
        self.dropout = nn.Dropout(dropout_rate)

        # Fully connected layers
        self.fc1 = nn.Linear(256 * 3 * 3, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 64)

    def forward_once(self, x):
        # Forward pass for one of the inputs
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2(x), 2))
        x = F.relu(F.max_pool2d(self.conv3(x), 2))
        x = F.relu(F.max_pool2d(self.conv4(x), 2))

        # Apply dropout after conv layers
        x = self.dropout(x)

        # Flatten the tensor before fully connected layers
        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)

        x = F.relu(self.fc2(x))
        x = self.dropout(x)

        x = F.relu(self.fc3(x))
        x = self.dropout(x)

        x = self.fc4(x)

        return x

    def forward(self, anchor, positive, negative):
        # Forward pass through the network for anchor, positive, and negative images
        anchor_output = self.forward_once(anchor)
        positive_output = self.forward_once(positive)
        negative_output = self.forward_once(negative)

        return anchor_output, positive_output, negative_output

#%% Triplet Cosine Loss

class TripletCosineLoss(nn.Module):
    def __init__(self, margin=0.5):
        super(TripletCosineLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        pos_dist = F.cosine_similarity(anchor, positive)
        neg_dist = F.cosine_similarity(anchor, negative)

        losses = F.relu(neg_dist - pos_dist + self.margin)
        return losses.mean()


#%% Train, eval and test data

data_train = {k:v for k, v in data_dict.items() if int(k.split('_')[0][1:]) <= 150}
data_eval = {k:v for k, v in data_dict.items() if 150 < int(k.split('_')[0][1:]) <200 }
data_test = {k:v for k, v in data_dict.items() if int(k.split('_')[0][1:]) > 200}


#%% Train, eval and test datasets and dataloaders

time_start = time.time()

# dataset_train = ImageDataset(data_train, num_pos=40000, num_neg=40000)
# dataset_eval = ImageDataset(data_eval, num_pos=15000, num_neg=15000)
# dataset_test = ImageDataset(data_test, num_pos=5000, num_neg=5000)

dataset_train = ImageDataset(data_train,num_data=20000)
dataset_eval = ImageDataset(data_eval, num_data=5000)
dataset_test = ImageDataset(data_test, num_data=5000)

batch_size = 64

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

print(f'Data loaded in {time.time() - time_start} sec')


#%% Hyperparameters

lr = 0.0001
weight_decay = 0 #.001# .0001 # .001 # .001 # 1e-3
num_epochs = 500
dropout = 0
margin = 0.1

#%%
### model initialization
model = SiameseTripletNetwork(dropout_rate=dropout)

# checkpoint = r"cnn_missing_images_overfitted_LossTrn_0.043425_AccTrn_98%_LossVal_2.110029_AccVal_63%.pt"
# model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

model.to(device)

print(f"Model parameters: {[p.numel() for p in model.parameters()]}")
print(f"Total parameters: {sum([p.numel() for p in model.parameters()])}")

### loss function
criterion = TripletCosineLoss(margin=margin)

num_train_steps = num_epochs * len(dataloader_train)

### optimizer with l2 regularization
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) ### L2 regularization
# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.3) ### L2 regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, cooldown=0)

#%%

losses_train, losses_val = [], []
accuracies_train, accuracies_val = [], []
loss_val_best = float('inf')

#%%

time_start = time.time()
progress_bar = tqdm(range(num_train_steps))

for epoch in range(num_epochs):

    model.train()
    loss_train_batch = 0
    correct_train, total_train = 0, 0
    conf_mat_all_trn = []

    for batch in dataloader_train:

        _, _, _, anchor, positive, negative = batch
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        optimizer.zero_grad()
        embed_anchor, embed_pos, embed_neg = model(anchor, positive, negative)
        loss = criterion(embed_anchor, embed_pos, embed_neg)
        loss.backward()
        optimizer.step()

        loss_train_batch += loss.item()

        if progress_bar is not None:
            progress_bar.update(1)

    ### Avg Loss over the epoch
    loss_train_avg = loss_train_batch / len(dataloader_train)
    losses_train.append(loss_train_avg)

    ### Validation loop
    model.eval()
    loss_val_batch = 0

    with torch.no_grad():
        _, _, _, anchor, positive, negative = batch
        anchor, positive, negative = anchor.to(device), positive.to(device), negative.to(device)

        embed_anchor, embed_pos, embed_neg = model(anchor, positive, negative)
        loss = criterion(embed_anchor, embed_pos, embed_neg)

        loss_val_batch += loss.item()

    ### Calculate average validation loss and accuracy
    loss_val_avg = loss_val_batch / len(dataloader_eval)
    losses_val.append(loss_val_avg)

    ### print the stats
    print(f"Epoch:{epoch + 1}/{num_epochs}, Loss_train:{loss_train_avg:.8f}, Loss_val:{loss_val_avg:.8f}")

    if loss_val_avg < loss_val_best:
        print(f"Saving model @ Loss_Val:{loss_val_avg:.4f}")


        torch.save({
            "checkpoint_initial": "Random Initialization",
            # "checkpoint_initial": checkpoint,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            "losses_train": losses_train,
            "losses_val": losses_val,

            "loss_val_best": loss_val_avg,
            "loss_at_train": loss_train_avg,

            "lr": lr,
            "dropout": dropout,
            "wt_decay": weight_decay,
            "batch_size": batch_size,
            'margin': margin,


        },
            "siamese_triplet_cnn4fc4.pt")

        loss_val_best = loss_val_avg

print("Finished Training")
print(f"Training time:{time.time() - time_start}")

# %% Visualize train and Test Loss

plt.figure(figsize=(20, 8))
plt.plot(range(len(losses_train)), losses_train, label='Train Loss', color='b', linestyle='-', linewidth=0.5,
         markersize=1)
plt.plot(range(len(losses_val)), losses_val, label='Test Loss', color='r', linestyle='-.', linewidth=0.5, markersize=1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and test loss curves')
plt.show()

# %% Rename the best file

loss_best_val = min(losses_val)
ind = losses_val.index(loss_best_val)
save_string = (f"_LossTrn_{losses_train[ind]:.6f}_LossVal_{losses_val[ind]:.6f}")
os.rename("siamese_triplet_cnn4fc4.pt", f"siamese_triplet_cnn4fc4{save_string}.pt")

# %% Save overfitted model

save_string = (f"_overfitted"
               f"_LossTrn_{loss_train_avg:.6f}_LossVal_{loss_val_avg:.6f}")

torch.save({
    "checkpoint_initial": "Random Initialization",
    # "checkpoint_initial": checkpoint,
    "epoch": epoch,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),

    "losses_train": losses_train,
    "losses_val": losses_val,

    "loss_val_best": loss_val_avg,
    "loss_at_train": loss_train_avg,

    "lr": lr,
    "dropout": dropout,
    "wt_decay": weight_decay,
    "batch_size": batch_size,
    'margin':margin,

},
    "siamese_triplet_cnn4fc4.pt")

os.rename("siamese_triplet_cnn4fc4.pt", f"siamese_triplet_cnn4fc4{save_string}.pt")
