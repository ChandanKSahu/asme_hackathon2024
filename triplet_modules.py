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

if __name__ == '__main__':
    model = SiameseTripletNetwork(dropout_rate=0)
    criterion = TripletCosineLoss(margin=0.3)

