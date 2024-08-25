import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from torchvision import transforms

from sklearn.metrics import confusion_matrix
from sklearn.manifold import TSNE

import os
import time
import pickle
from tqdm.auto import tqdm

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import random

from triplet_modules import SiameseTripletNetwork, TripletCosineLoss, ImageDataset

#%% Torch device

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(torch.cuda.get_device_properties(device))

#%% Data Loading

nist_part1 = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\Codes\hack2024\nist_eval_part1.pkl"
nist_part2 = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\Codes\hack2024\nist_eval_part2.pkl"

time_start = time.time()

with open(nist_part1, 'rb') as f:
    data_dict = pickle.load(f)

print(f"Data read in {time.time() - time_start} seconds.")

#%%

class ImageDatasetNIST(Dataset):
    def __init__(self, data_dict):
        self.data_dict = data_dict
        self.data = dict(sorted(data_dict.items()))
        self.triplets = self._create_triplets()

    def _create_triplets(self):
        triplets = []
        filenames = list(self.data_dict.keys())

        for i in range(1, len(filenames)-2):
            fname_prv, fname_cur, fname_nxt = filenames[i], filenames[i+1], filenames[i+2]

            if fname_prv +1 == fname_cur == fname_nxt -1:

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


#%%

dataset_test = ImageDatasetNIST(data_dict)
batch_size = 1
dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=False)

#%% Model Loading

model = SiameseTripletNetwork(dropout_rate=0)

checkpoint_loc = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\Codes\hack2024\siamese_triplet_cnn4fc4_trn_0.008037_val_0.000005.pt"
checkpoint = torch.load(checkpoint_loc)

# checkpoint_loc2 = r"E:\OneDrive\OneDrive - Clemson University\Important\Thesis\Codes\hack2024\siamese_triplet_cnn4fc4_LossTrn_0.006602_LossVal_0.000004.pt"

model.load_state_dict(checkpoint['model_state_dict'])
model.to(device)

#%% Loss Loading

criterion = TripletCosineLoss(margin=0.3)

#%% Evaluation - total loss

model.eval()
loss_val_batch = 0
# sim_pos_all, sim_neg_all = {}, {}
sim_pos_neg = {}
embeddings_all = {}

with torch.no_grad():
    for batch in dataloader_test:
        fname_anchor, fname_pos, fname_neg, anchor, positive, negative = batch
        anchor = anchor.to(device)
        positive = positive.to(device)
        negative = negative.to(device)

        embed_anchor, embed_pos, embed_neg = model(anchor, positive, negative)
        loss = criterion(embed_anchor, embed_pos, embed_neg)

        sim_pos = F.cosine_similarity(embed_anchor, embed_pos)
        sim_neg = F.cosine_similarity(embed_anchor, embed_neg)

        loss_val_batch += loss.item()

        sim_pos_neg[fname_anchor] = (sim_pos, sim_neg)
        embeddings_all[fname_anchor] = embed_anchor

    ### Calculate average validation loss and accuracy
    loss_val_avg = loss_val_batch / len(dataloader_test)

print("Loss val Avg: ", loss_val_avg)

sim_pos_neg = dict(sorted(sim_pos_neg.items()))
embeddings_all = dict(sorted(embeddings_all.items()))


#%%
sim_pos_all = [v[0] for v in sim_pos_neg.values()]
sim_pos_all = torch.cat(sim_pos_all, 0).cpu().numpy()

sim_neg_all = [v[1] for v in sim_pos_neg.values()]
sim_neg_all = torch.cat(sim_neg_all, 0).cpu().numpy()

#%% Hexbin plot of similarities

# Scatter plot with alpha blending to show density
plt.figure(figsize=(8, 6))
plt.hexbin(sim_pos_all, sim_neg_all, gridsize=50, cmap='inferno', bins='log')
plt.colorbar(label='log10(N)')
plt.title('Density Plot of #ImageTriplets')
plt.xlabel('Positive Similarity')
plt.ylabel('Negative Similarity')
plt.show()

#%% Contour plot

# Create a 2D histogram
plt.figure(figsize=(8, 6))
hist, xedges, yedges = np.histogram2d(sim_pos_all, sim_neg_all, bins=100)

# Compute the x and y midpoints of the bins
xmid = (xedges[:-1] + xedges[1:]) / 2
ymid = (yedges[:-1] + yedges[1:]) / 2

# Plot the contours
contourf = plt.contourf(xmid, ymid, hist.T, levels=20, cmap='inferno')
contour = plt.contour(xmid, ymid, hist.T, levels=5, colors='black', linewidths=0.5)

plt.colorbar(contourf, label='Density')
plt.title('Contour Plot of #ImageTriplets')
plt.xlabel('Positive Similarity')
plt.ylabel('Negative Similarity')
plt.show()

#%% Histogram Plots

plt.figure()

# Create the histogram with KDE curve for the first variable
sns.histplot(sim_pos_all, bins=500, kde=True, color='blue', edgecolor='black', label='Negative Similarity', stat="density")

# Create the histogram with KDE curve for the second variable
sns.histplot(sim_neg_all, bins=500, kde=True, color='red', edgecolor='black', label='Positive Similarity', stat="density")

# Add titles and labels
plt.title('Histograms with KDE Curves')
plt.xlabel('Value')
plt.ylabel('Density')
plt.legend()
plt.show()

#%%

from sklearn.metrics import roc_curve, auc

def plot_roc_curve(sim_pos, sim_neg):
    # Create labels for positive (1) and negative (0)
    y_true = [1] * len(sim_pos) + [0] * len(sim_neg)
    y_scores = list(sim_pos) + list(sim_neg)

    # Compute ROC curve and AUC
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)
    roc_auc = auc(fpr, tpr)

    plt.figure(figsize=(8, 8))
    plt.plot(fpr, tpr, color='b', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='r', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    plt.show()

plot_roc_curve(sim_pos_all, sim_neg_all)

#%% Embeddings all

embedd_anc = [v[0].unsqueeze(0) for v in embeddings_all.values()]
embedd_anc = torch.cat(embedd_anc, dim=0).cpu().numpy()
#
#%% TSNE CPU - SkLearn

rand_state = 5

embed_tsne = TSNE(
    n_components=2, ### Dimension of the embedded space - 2
    perplexity=2, # len(req_list)-1,  ### number of neighbors for learning the manifold - 30
    early_exaggeration=12, ### looseness of the clusters - high => loose - 12
    learning_rate='auto', ### [10,1000] high=>equidistant from neighbors(~ball), low => point like - auto=max(N/early_exaggeration/4, 50)
    n_iter=1000, ### #iterations in [250, 1000]
    min_grad_norm=1e-6, ### convergence limit - 1e-7
    metric='cosine', # 'euclidean' 'cosine',
    ### https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.pdist.html
    ### ‘braycurtis’, ‘canberra’, ‘chebyshev’, ‘cityblock’, ‘correlation’, ‘cosine’, ‘dice’, ‘euclidean’, ‘hamming’,
    ### ‘jaccard’, ‘jensenshannon’, ‘kulczynski1’, ‘mahalanobis’, ‘matching’, ‘minkowski’, ‘rogerstanimoto’,
    ### ‘russellrao’, ‘seuclidean’, ‘sokalmichener’, ‘sokalsneath’, ‘sqeuclidean’, ‘yule’
    ### https://scikit-learn.org/stable/modules/generated/sklearn.metrics.pairwise.distance_metrics.html
    ### ‘haversine’, 'manhattan', 'l1', 'l2', 'nan_euclidean'

    init='pca',  ### 'pca'/'random' - Initialization of the embedding
    method='exact', ### default:'barnes_hut'
    ### Use 'exact' if the nearest neighbor errors shall be <3%
    verbose=1,
    random_state=rand_state,
)

embed_tsne.fit_transform(embedd_anc)
print(f"KL Divergence: {embed_tsne.kl_divergence_}")
print(f"n_features_in: {embed_tsne.n_features_in_}")
print(f"Learning rate: {embed_tsne.learning_rate_}")
print(f"n_iter_: {embed_tsne.n_iter_}")

#%%

tsne_embed = embed_tsne.embedding_

###%% Visualize t-SNE embeddings
import matplotlib
# matplotlib.use('Qt5Agg')
sns.set_style('whitegrid')
# sns.reset_orig()

# plt.figure(figsize=(15,15), dpi=300)
plt.figure()
plt.scatter(tsne_embed[:, 0], tsne_embed[:,1],
            # c= req_df['Color'], # 'cornsilk', # marker color
            # s=100, ## marker size
            # linewidths=0.5, # linewidth of marker edges
            # edgecolors='k' ## edge color of the marker
            )

plt.savefig('pretrained_t_sne_plt.png')
plt.show()
# plt.close()
