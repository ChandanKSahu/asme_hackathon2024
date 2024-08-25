import os
from PIL import Image
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms

from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

from triplet_modules import SiameseTripletNetwork, TripletCosineLoss

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

#%% Read NIST Data

def load_images_to_dict(folder_path) -> dict:

    image_dict = {}

    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        image = Image.open(file_path)

        image_array = np.array(image)
        image_array = cv2.resize(image_array, (0, 0), fx=0.5, fy=0.5)

        image_dict[int(file_name.split(".")[0][-6:])] = image_array

    return image_dict

folder_path_p1 = r'E:\Hackathon2024Data\NIST\NIST_Problem_full_dataset\opening_release_part1\part1_given'
folder_path_p1_miss = r'E:\Hackathon2024Data\NIST\NIST_Problem_full_dataset\opening_release_part1\part1_missing'

images_p1 = load_images_to_dict(folder_path_p1)
images_p1_miss = load_images_to_dict(folder_path_p1_miss)

# with open("nist_eval_part1.pkl", 'wb') as file:
#     pickle.dump(images_p1, file)

#%% Dataset

class ImageDataset(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.data = dict(sorted(data_dict.items()))
        self.triplets = self._create_triplets()

    def _create_triplets(self):
        triplets = []
        filenames = list(self.data_dict.keys())

        for i in range(1, len(filenames)-1, 1):

            fname_cur, fname_nxt = filenames[i], filenames[i+1]

            if fname_cur == fname_nxt - 1:

                    data_cur = torch.from_numpy(self.data[fname_cur] / 255).float().unsqueeze(0)
                    data_nxt = torch.from_numpy(self.data[fname_nxt] / 255).float().unsqueeze(0)

                    triplets.append((
                        fname_cur, fname_nxt, data_cur, data_nxt
                    ))
        return triplets

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        return self.triplets[idx]

#%%

dataset_nist_p1 = ImageDataset(images_p1)

batch_size = 1

dataloader_nist_p1 = DataLoader(dataset_nist_p1, batch_size=batch_size, shuffle=False)


#%% Model Loading

model = SiameseTripletNetwork(dropout_rate=0)

checkpoint_loc = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\Codes\hack2024\siamese_triplet_cnn4fc4_trn_0.008037_val_0.000005.pt"
# checkpoint_loc = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\Codes\hack2024\siamese_triplet_cnn4fc4_LossTrn_0.003654_LossVal_0.000001.pt"
checkpoint = torch.load(checkpoint_loc)

# checkpoint_loc2 = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\Codes\hack2024\siamese_triplet_cnn4fc4_LossTrn_0.006602_LossVal_0.000004.pt"

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)


#%% Evaluation - total loss

model.eval()

sim_all_p1 = {}
embeddings_all_p1 = {}

with torch.no_grad():
    for batch in dataloader_nist_p1:
        fname_cur, fname_nxt, frame_cur, frame_nxt = batch
        frame_cur = frame_cur.to(device)
        frame_nxt = frame_nxt.to(device)

        embed_cur = model.forward_once(frame_cur)
        embed_nxt = model.forward_once(frame_nxt)

        sim = F.cosine_similarity(embed_cur, embed_nxt)

        sim_all_p1[fname_cur.item()] = sim.item()
        embeddings_all_p1[fname_cur.item()] = embed_cur

#%% Locations of missing frames

sim_df_p1 = pd.DataFrame.from_dict(sim_all_p1, orient='index')

threshold = 0.272
# threshold = 0.24

sim_filtered_p1 = sim_df_p1[sim_df_p1[0] < threshold]
miss_frm_loc = sim_filtered_p1[0].to_dict()

#%% Verification

cos_sim = {}

for index, miss_frame_num in enumerate(miss_frm_loc.keys(), start=1):

    frame_miss = images_p1_miss[index]
    frame_prv = images_p1[miss_frame_num]
    frame_nxt = images_p1[miss_frame_num +1]

    frame_miss = torch.from_numpy(frame_miss / 255).float().unsqueeze(0).unsqueeze(0).to(device)
    frame_prv = torch.from_numpy(frame_prv / 255).float().unsqueeze(0).unsqueeze(0).to(device)
    frame_nxt = torch.from_numpy(frame_nxt / 255).float().unsqueeze(0).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        embed_miss, embed_prv, embed_nxt = model(frame_miss, frame_cur, frame_nxt)

        cos_miss_prv = F.cosine_similarity(embed_miss, embed_prv)
        cos_miss_nxt = F.cosine_similarity(embed_miss, embed_nxt)
        cos_prv_nxt = F.cosine_similarity(embed_prv, embed_nxt)
        combine = cos_miss_prv * cos_miss_nxt / cos_prv_nxt

        cos_sim[miss_frame_num] = (cos_miss_prv.item(), cos_miss_nxt.item(), cos_prv_nxt.item(), combine.item())

cos_sim_df = pd.DataFrame.from_dict(cos_sim, orient='index')


#%% Deep Search

# miss_dataset = ImageDataset()
#
# for frm_miss in images_p1_miss.values():
#
#     frame_miss = torch.from_numpy(frm_miss / 255).float().unsqueeze(0)
#     frame_miss = frame_miss.to(device)
#
#     model.eval()
#
#     with torch.no_grad():
#         for batch in dataloader_nist_p1:
#             fname_cur, fname_nxt, frame_cur, frame_nxt = batch
#             frame_cur = frame_cur.to(device)
#             frame_nxt = frame_nxt.to(device)
#
#             embed_miss = model.forward_once(frame_miss)
#             embed_cur = model.forward_once(frame_cur)
#             embed_nxt = model.forward_once(frame_nxt)
#
#             sim_cur_nxt = F.cosine_similarity(embed_cur, embed_nxt)
#             sim_miss_cur = F.cosine_similarity(embed_miss, embed_cur)
#             sim_miss_nxt = F.cosine_similarity(embed_miss, embed_nxt)
#
#             common = sim_miss_cur * sim_miss_nxt / sim_cur_nxt
#
#             sim_all_p1[fname_cur[0].item()] = sim.item()
#             embeddings_all_p1[fname_cur[0].item()] = embed_cur
#



#%% Part 2

data_loc_p2 = r"E:\Hackathon2024Data\NIST\NIST_Problem_full_dataset\opening_release_part2\part2_given"

images_p2 = load_images_to_dict(data_loc_p2)

dataset_nist_p2 = ImageDataset(images_p2)
dataloader_nist_p2 = DataLoader(dataset_nist_p2, batch_size=batch_size, shuffle=False)

model.eval()

sim_all_p2 = {}
embeddings_all_p2 = {}

with torch.no_grad():
    for batch in dataloader_nist_p2:
        fname_cur, fname_nxt, frame_cur, frame_nxt = batch
        frame_cur = frame_cur.to(device)
        frame_nxt = frame_nxt.to(device)

        embed_cur = model.forward_once(frame_cur)
        embed_nxt = model.forward_once(frame_nxt)

        sim = F.cosine_similarity(embed_cur, embed_nxt)

        sim_all_p2[fname_cur.item()] = sim.item()
        embeddings_all_p2[fname_cur.item()] = embed_cur

sim_df_p2 = pd.DataFrame.from_dict(sim_all_p2, orient='index')

threshold = -0.501
# threshold = -0.555

sim_filtered_p2 = sim_df_p2[sim_df_p2[0]<threshold]

#%%

