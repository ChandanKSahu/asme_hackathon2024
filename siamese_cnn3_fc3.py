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
    def __init__(self, data_dict, num_pos=50000, num_neg=50000):

        self.data_dict = data_dict
        self.num_pos = num_pos
        self.num_neg = num_neg

        self.positive_pairs = self._create_positive_pairs()
        self.negative_pairs = self._create_negative_pairs()

        print(f"Pos pairs={len(self.positive_pairs)}\nNeg Pairs={len(self.negative_pairs)}")

        self.pairs = self.positive_pairs + self.negative_pairs

    def _create_positive_pairs(self):
        """
        Creates positive pairs:
        Images are consecutive.
        Belong to the same layer. The difference between their indices is 1
        :return:
        """
        pos_pairs = []

        file_names = list(self.data_dict.keys())

        for i in range(len(self.data_dict)-1):
            fname_cur, fname_nxt = file_names[i], file_names[i+1]

            layer_cur, frame_cur = fname_cur.split("_")
            layer_nxt, frame_nxt = fname_nxt.split("_")

            if layer_cur==layer_nxt and int(frame_nxt[1:])-int(frame_cur[1:])==1:
                diff = self.data_dict[fname_cur] - self.data_dict[fname_nxt]

                # image_3c = np.stack((self.data_dict[fname_cur], diff, self.data_dict[fname_nxt]), axis=0)
                # image_3c = torch.from_numpy(image_3c/255).float()

                image_cur = torch.from_numpy(self.data_dict[fname_cur]/255).unsqueeze(0).float()
                image_nxt = torch.from_numpy(self.data_dict[fname_nxt]/255).unsqueeze(0).float()

                pos_pairs.append((fname_cur, fname_nxt, 1, image_cur, image_nxt))
                # pos_pairs.append((fname_cur, fname_nxt, 1, image_3c))

        pos_samples = random.sample(pos_pairs, self.num_pos)
        return pos_samples

    def _create_negative_pairs(self):
        """
        Creates negative pairs:
        Only one frame is missing between the pair of images
        Belong to the same layer. The difference between their indices is 2.

        :return:
        """
        neg_pairs = []

        file_names = list(self.data_dict.keys())

        for i in range(len(self.data_dict)-2):
            fname_cur, fname_nxt = file_names[i], file_names[i+2]

            layer_cur, frame_cur = fname_cur.split("_")
            layer_nxt, frame_nxt = fname_nxt.split("_")

            if layer_cur == layer_nxt and int(frame_nxt[1:])-int(frame_cur[1:])==2:
                diff = self.data_dict[fname_cur] - self.data_dict[fname_nxt]

                # image_3c = np.stack((self.data_dict[fname_cur], diff, self.data_dict[fname_nxt]), axis=0)
                # image_3c = torch.from_numpy(image_3c/255).float()

                image_cur = torch.from_numpy(self.data_dict[fname_cur] / 255).unsqueeze(0).float()
                image_nxt = torch.from_numpy(self.data_dict[fname_nxt] / 255).unsqueeze(0).float()

                neg_pairs.append((fname_cur, fname_nxt, 0, image_cur, image_nxt))
                # neg_pairs.append((fname_cur, fname_nxt, 0, image_3c))

        neg_samples = random.sample(neg_pairs, self.num_neg)
        return neg_samples

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        return self.pairs[idx]

#%% Create Neural Network
class SiameseNetwork(nn.Module):
    def __init__(self, dropout_prob=0.5):
        super(SiameseNetwork, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)  # Same padding for 3x3 kernel
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)  # Same padding for 3x3 kernel
        self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)  # Same padding for 3x3 kernel
        self.fc1 = nn.Linear(128 * 7 * 7, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.dropout = nn.Dropout(dropout_prob)

    def forward_once(self, x):
        x = F.relu(self.conv1(x))  # 60x60 -> 60x60
        x = F.max_pool2d(x, 2)      # 60x60 -> 30x30

        x = F.relu(self.conv2(x))  # 30x30 -> 30x30
        x = F.max_pool2d(x, 2)      # 30x30 -> 15x15

        x = F.relu(self.conv3(x))  # 15x15 -> 15x15
        x = F.max_pool2d(x, 2)      # 15x15 -> 7x7

        x = x.view(x.size()[0], -1)  # Flatten the tensor
        x = F.relu(self.fc1(x))

        x = self.dropout(x)         # Apply dropout
        x = F.relu(self.fc2(x))

        x = self.dropout(x)         # Apply dropout
        x = self.fc3(x)

        return x

    def forward(self, input1, input2):
        output1 = self.forward_once(input1)
        output2 = self.forward_once(input2)
        return output1, output2


#%% loss function

class ContrastiveLossCosine(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLossCosine, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        # Normalize embeddings
        output1 = F.normalize(output1, p=2, dim=1)
        output2 = F.normalize(output2, p=2, dim=1)

        # Compute cosine similarity
        cosine_similarity = F.cosine_similarity(output1, output2)

        # Compute cosine distance
        cosine_distance = 1 - cosine_similarity

        # Contrastive loss
        loss = torch.mean((1 - label) * torch.pow(cosine_distance, 2) +
                          (label) * torch.pow(torch.clamp(self.margin - cosine_distance, min=0.0), 2))
        return loss

class ContrastiveLoss(nn.Module):
    def __init__(self, margin=1.0):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin

    def forward(self, output1, output2, label):
        euclidean_distance = F.pairwise_distance(output1, output2)
        loss_contrastive = torch.mean((1 - label) * torch.pow(euclidean_distance, 2) +
                                      (label) * torch.pow(torch.clamp(self.margin - euclidean_distance, min=0.0), 2))
        return loss_contrastive

#%% Train, eval and test data
data_test = {k:v for k, v in data_dict.items() if int(k.split('_')[0][1:]) in [248, 249, 250]}
data_eval = {k:v for k, v in data_dict.items() if 200< int(k.split('_')[0][1:]) <248 }
data_train = {k:v for k, v in data_dict.items() if int(k.split('_')[0][1:]) < 200}

#%% Train, eval and test datasets and dataloaders

time_start = time.time()

# dataset_train = ImageDataset(data_train, num_pos=40000, num_neg=40000)
# dataset_eval = ImageDataset(data_eval, num_pos=10000, num_neg=10000)
# dataset_test = ImageDataset(data_test, num_pos=5000, num_neg=5000)

dataset_train = ImageDataset(data_train, num_pos=8000, num_neg=8000)
dataset_eval = ImageDataset(data_eval, num_pos=2500, num_neg=2500)
dataset_test = ImageDataset(data_test, num_pos=500, num_neg=500)

batch_size = 64

dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataloader_eval = DataLoader(dataset_eval, batch_size=batch_size, shuffle=True)
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

print(f'Data loaded in {time.time() - time_start} sec')

#%% Hyperparameters

lr = 0.001
weight_decay = 0 #.001# .0001 # .001 # .001 # 1e-3
num_epochs = 50
dropout = 0.1
threshold = 0.4

#%%
### model initialization
model = SiameseNetwork(dropout_prob=dropout)

# checkpoint = r"cnn_missing_images_overfitted_LossTrn_0.043425_AccTrn_98%_LossVal_2.110029_AccVal_63%.pt"
# model.load_state_dict(torch.load(checkpoint)['model_state_dict'])

model.to(device)

print(f"Model parameters: {[p.numel() for p in model.parameters()]}")
print(f"Total parameters: {sum([p.numel() for p in model.parameters()])}")

# criterion = ContrastiveLoss(margin=threshold)
criterion = ContrastiveLossCosine(margin=threshold)

num_train_steps = num_epochs * len(dataloader_train)

### optimizer with l2 regularization
optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay) ### L2 regularization
# optimizer = optim.SGD(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=0.3) ### L2 regularization
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=10, cooldown=0)

#%%
losses_train, losses_val = [], []
accuracies_train, accuracies_val = [], []
loss_val_best = float('inf')

#%% Training

time_start = time.time()
progress_bar = tqdm(range(num_train_steps))

for epoch in range(num_epochs):

    model.train()
    loss_train_batch = 0
    correct_train, total_train = 0, 0
    conf_mat_all_trn = []

    for batch in dataloader_train:
        fname_cur, fname_nxt, label, image1, image2 = batch

        image1 = image1.to(device)
        image2 = image2.to(device)
        label = label.to(device)

        optimizer.zero_grad()
        output1, output2 = model(image1, image2)
        loss = criterion(output1, output2, label)
        loss.backward()
        optimizer.step()

        loss_train_batch += loss.item()

        # Compute distance (assuming criterion is contrastive loss)
        cosine_distance = F.cosine_similarity(output1, output2)
        predicted = (cosine_distance > threshold).float()

        total_train += label.size(0)  # Total number of instances
        correct_train += (predicted == label).sum().item()  # Count of correct predictions
        conf_mat = confusion_matrix(label.cpu().numpy(), predicted.cpu().numpy())
        conf_mat_all_trn.append(conf_mat)

        if progress_bar is not None:
            progress_bar.update(1)

    ### Avg Loss over the epoch
    loss_train_avg = loss_train_batch / len(dataloader_train)
    losses_train.append(loss_train_avg)

    accuracy_train = 100 * correct_train / total_train
    accuracies_train.append(accuracy_train)

    ### Validation loop
    model.eval()
    loss_val_batch = 0
    correct_eval, total_eval = 0, 0
    conf_mat_all_val = []

    with torch.no_grad():
        for batch in dataloader_eval:
            fname_cur, fname_nxt, label, image1, image2 = batch
            image1 = image1.to(device)
            image2 = image2.to(device)
            label = label.to(device)
            output1, output2 = model(image1, image2)
            loss = criterion(output1, output2, label)

            loss_val_batch += loss.item()

            # Compute distance (assuming criterion is contrastive loss)
            cosine_distance = F.cosine_similarity(output1, output2)
            predicted = (cosine_distance > threshold).float()
            total_eval += label.size(0) # Total number of instances

            conf_mat = confusion_matrix(label.cpu().numpy(), predicted.cpu().numpy())
            conf_mat_all_val.append(conf_mat)

    ### Calculate average validation loss and accuracy
    loss_val_avg = loss_val_batch / len(dataloader_eval)
    losses_val.append(loss_val_avg)

    scheduler.step(loss_val_avg)

    accuracy_eval = 100 * correct_eval / total_eval
    accuracies_val.append(accuracy_eval)

    ### print the stats
    print(f"Epoch:{epoch+1}/{num_epochs}, Loss_train:{loss_train_avg:.6f}, Loss_val:{loss_val_avg:.6f} "
          f" Acc_Train:{accuracy_train:.2f}% Acc_Val:{accuracy_eval:.2f}%")

    if loss_val_avg < loss_val_best:
        print(f"Saving model @ Loss_Val:{loss_val_avg:.4f} Acc_val:{accuracy_eval:.2f}%")

        ### Get the confusion matrix of the complete validation dataset
        conf_mat_total_val = np.sum(np.stack(conf_mat_all_val, axis=0), axis=0)
        print(f"Confusion Matrix Validation:\n{conf_mat_total_val}")
        conf_mat_total_trn = np.sum(np.stack(conf_mat_all_trn, axis=0), axis=0)
        print(f"Confusion Matrix Training:\n{conf_mat_total_trn}")

        torch.save({
            "checkpoint_initial": "Random Initialization",
            # "checkpoint_initial": checkpoint,
            "epoch":epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            "losses_train": losses_train,
            "losses_val": losses_val,
            "accuracies_train": accuracies_train,
            "accuracies_val": accuracies_val,

            "loss_val_best": loss_val_avg,
            "loss_at_train": loss_train_avg,
            "accuracy_train": accuracy_train,
            "accuracy_val": accuracy_eval,

            "lr": lr,
            "dropout": dropout,
            "wt_decay":weight_decay,
            "batch_size": batch_size,

        },
        "cnn_missing_images.pt")

        loss_val_best = loss_val_avg

print("Finished Training")
print(f"Training time:{time.time()-time_start}")

#%% Visualize train and Test Loss

plt.figure(figsize=(20,8))
plt.plot(range(len(losses_train)), losses_train, label='Train Loss', color='b', linestyle='-', linewidth=0.5, markersize=1)
plt.plot(range(len(losses_val)), losses_val, label='Test Loss', color='r', linestyle='-.', linewidth=0.5, markersize=1)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.title('Training and test loss curves')
plt.show()

#%% Rename the best file

loss_best_val = min(losses_val)
ind = losses_val.index(loss_best_val)
save_string = (f"_LossTrn_{losses_train[ind]:.6f}_AccTrn_{accuracies_train[ind]:.0f}%"
               f"_LossVal_{losses_val[ind]:.6f}_AccVal_{accuracies_val[ind]:.0f}%")
os.rename("cnn_missing_images.pt", f"cnn_missing_images{save_string}.pt")

#%% Save overfitted model

save_string = (f"_overfitted"
               f"_LossTrn_{loss_train_avg:.6f}_AccTrn_{accuracy_train:.0f}%"
               f"_LossVal_{loss_val_avg:.6f}_AccVal_{accuracy_eval:.0f}%")

torch.save({
            "checkpoint_initial": "Random Initialization",
            # "checkpoint_initial": checkpoint,
            "epoch":epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),

            "losses_train": losses_train,
            "losses_val": losses_val,
            "accuracies_train": accuracies_train,
            "accuracies_val": accuracies_val,

            "loss_val_best": loss_val_avg,
            "loss_at_train": loss_train_avg,
            "accuracy_train": accuracy_train,
            "accuracy_val": accuracy_eval,

           "lr": lr,
            "dropout": dropout,
            "wt_decay":weight_decay,
            "batch_size": batch_size,

        },
        "cnn_missing_images.pt")

os.rename("cnn_missing_images.pt", f"cnn_missing_images{save_string}.pt")
