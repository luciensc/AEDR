"""
author: Lucien Schläpfer
date: 27.08.2020

research question: is it possible to obtain a robust 2D representation using an autoencoder
and noise injections at various stages
- during each epoch before feeding into model
- in the 2D representation

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
from run import run_aedr
num_epochs = 300
loss2_coef = 0  # 1e-3  # weight on penalty of z0 distribution

# noise cannot be 0, but arbitrarily small.
ls_noise_pre = [0.025, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6]
ls_noise_z0 = [0.025, 0.05, 0.1, 0.15, 0.2, 0.4, 0.6]
n_reps = 10


folder =  "datasets/"
outdir = "model_output/"
runplot = False  # show train & val loss, and L1 dist matrix loss across epochs for each screened param combination

###
dat = np.genfromtxt(folder+"X_1000_circle5Dlin.csv", delimiter=",")
#y = np.genfromtxt(folder+"y_1000_circle.csv", delimiter=",")

#dat_train, dat_val, y_train, y_val = train_test_split(dat, y)
dat_train, dat_val = train_test_split(dat)
dist_pre = pairwise_distances(dat_val)
val_loss_mtrx = np.full((len(ls_noise_pre), len(ls_noise_z0)), np.nan)
val_distl1_mtrx = np.full((len(ls_noise_pre), len(ls_noise_z0)), np.nan)

for i_pre, noise_pre in enumerate(ls_noise_pre):
    print("noise pre: ", noise_pre)
    for j_z0, noise_z0 in enumerate(ls_noise_z0):
        print("\tnoise_z0:", noise_z0)
        ls_valloss = []
        ls_distL1 = []
        for seed in range(n_reps):
            runlog = run_aedr(seed=seed, dat_train=dat_train, dat_val=dat_val, noise_pre=noise_pre, noise_z0=noise_z0,
                              dist_pre=dist_pre)
            ls_valloss.append(np.min(runlog['loss1_val']))
            ls_distL1.append(np.min(runlog['distL1val']))

            if runplot:
                plt.figure(figsize=(10, 5))
                plt.plot(runlog["epochs"], runlog["loss1_train"], "b-.", label="loss1 train")
                plt.plot(runlog["epochs"], runlog["loss1_val"], "b-", label="loss1 val")
                plt.plot(runlog["epochs"], runlog["distL1val"], "r--", label="dist L1 val")
                plt.ylim(0,3)
                plt.legend()
                plt.show()
                plt.close()

        print(f"\t\tval loss: {np.mean(ls_valloss):.2f} +- {np.std(ls_valloss):.2f}")
        print(f"\t\tl1 matrix loss: {np.mean(ls_distL1):.2f} +- {np.std(ls_distL1):.2f}")
        val_loss_mtrx[i_pre, j_z0] = np.mean(ls_valloss)
        val_distl1_mtrx[i_pre, j_z0] = np.mean(ls_distL1)

# TODO (low priority): imprecise labelling: label ought be at centre of corresponding cell, not at the border.
# switched axes bc indexed in 'matrix format'!!
plt.pcolormesh([0]+ls_noise_z0, [0]+ls_noise_pre, val_loss_mtrx, cmap=cm.Reds)
plt.xlabel("noise z0")
plt.ylabel("noise pre")
plt.title("val loss")
plt.colorbar()
plt.clim(vmin=0)
plt.show()
plt.close()

plt.pcolormesh([0]+ls_noise_z0, [0]+ls_noise_pre, val_distl1_mtrx, cmap=cm.Reds)
plt.xlabel("noise z0")
plt.ylabel("noise pre")
plt.title("val dist. matrix L1 norm")
plt.colorbar()
plt.clim(vmin=0)
plt.show()