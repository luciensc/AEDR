"""
perform one run for a particular parameter combination
"""


import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.metrics.pairwise import pairwise_distances
from torch.utils.data import DataLoader
import matplotlib
#matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import cm

from utils import *

#######################################################################################################################
def run_aedr(seed, dat_train, dat_val, noise_pre, noise_z0, dist_pre, num_epochs=300):
    np.random.seed(seed)
    torch.manual_seed(seed)
    dl = DataLoader(dat_train, batch_size=5000, shuffle=True)

    model = AEDR(input_dim=dat_train.shape[1], n_enc_layers=1, n_dec_layers=1, n_units=50, noise=noise_z0)
    mse_loss = nn.MSELoss()
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)

    ### START TRAINING
    log = pd.DataFrame(np.full((num_epochs, 4), np.nan), columns=["epochs", "loss1_train", "loss1_val", "distL1val"])
    model.train()
    #print("epoch\t\tloss\tloss1\tloss2\tloss1 VAL")
    for epoch in range(0, num_epochs):
        loss_tally = 0
        loss_1_tally = 0
        loss_2_tally = 0
        for idx, X in enumerate(dl):
            optim.zero_grad()
            X = X.float() + torch.normal(0, noise_pre, size=X.size())
            z0, X_pred = model.forward(X)

            # loss: 2 components: MSE(X, X_pred); (z0 deviation from origin (e.g. KL loss between normal(0,1) and normal(mean(z0), std(z0)) )
            loss1 = mse_loss(X, X_pred)
            loss2 = l2loss(z0.reshape(-1))
            # loss2 = gaussian_KL(mu_1=torch.mean(z0.reshape(-1)), mu_2=torch.tensor(0, dtype=torch.float32),
            #                     sigma_1=torch.std(z0.reshape(-1)), sigma_2=torch.tensor(1, dtype=torch.float32))
            loss = loss1 #+ loss2_coef * loss2
            loss.backward()
            optim.step()
            loss_tally += loss.detach().item()/X.shape[0]*1000
            loss_1_tally += loss1.detach().item()/X.shape[0]*1000
            loss_2_tally += loss2.detach().item()/X.shape[0]*1000 #* loss2_coef

        # calc val error
        X_red, X_pred = model.forward(torch.tensor(dat_val, dtype=torch.float32))
        loss1_val = mse_loss(torch.tensor(dat_val, dtype=torch.float32), X_pred).detach().item()/X_red.shape[0]*1000
        dist_post = pairwise_distances(X_red.detach())
        dist_l1 = matrix_L1(dist_pre, dist_post)
        # print(f"epoch {epoch}\t\t{loss_tally:.2f}\t{loss_1_tally:.2f}\t{loss_2_tally:.2f}\t{loss1_val:.2f}")
        # print(f"distance matrix L1 norm: {dist_l1:2f}")
        log.iloc[epoch] = [epoch, loss_1_tally, loss1_val, dist_l1]

        # # plot 2D representation at some epoch interval. a.o. requires modification to function arguments to pass labels.
        # if (epoch % 100 == 0):
        #     X_red = X_red.detach().numpy()
        #     # X_red_labelled = X_red[is_not_nan, :]
        #
        #     plt.figure(figsize=(8, 8))
        #     plt.scatter(X_red[:, 0], X_red[:, 1], c=y_val, alpha=0.3)
        #     # plt.xlim(-lim, lim)
        #     # plt.ylim(-lim, lim)
        #     plt.xlabel("dim 1")
        #     plt.ylabel("dim 2")
        #     # plt.legend()
        #     # if epoch == num_epochs
        #     # plt.savefig(outdir + "AEDR_cross_patient"+str(epoch)+".png")
        #     plt.close()

    return log
