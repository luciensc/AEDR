import torch
import torch.nn as nn

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.preprocessing import StandardScaler

n_samples = 1000
outdir = "datasets/"


# 2D datasets generated following the sklearn.cluster doc page
# (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html, 2.8.2020)

### generate 2D data in several clusters; assign labels
np.random.seed(0)
blobs = datasets.make_blobs(n_samples=n_samples, random_state=8)
noisy_moons = datasets.make_moons(n_samples=n_samples, noise=.05)
noisy_circles = datasets.make_circles(n_samples=n_samples, factor=.5,
                                      noise=.05)
# anisotropic gaussians
random_state = 170
X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
transformation = [[0.6, -0.6], [-0.4, 0.8]]
X_aniso = np.dot(X, transformation)
aniso = (X_aniso, y)

# blobs with varied variances
varied = datasets.make_blobs(n_samples=n_samples,
                             cluster_std=[1.0, 2.5, 0.5],
                             random_state=random_state)

### define linear and non-linear transformations
np.random.seed(33)
torch.manual_seed(66)
# linear:
A5 = np.random.uniform(-1, 1, (2, 5))
A20 = np.random.uniform(-1, 1, (2, 20))
# MLP
MLP5 = nn.Sequential(nn.Linear(2,50), nn.Sigmoid(), nn.Linear(50, 5))
MLP20 = nn.Sequential(nn.Linear(2,50), nn.Sigmoid(), nn.Linear(50, 20))

# plot, transform, save
ds_dict = {"gaussians":blobs, "crescents":noisy_moons, "circle":noisy_circles, "anisotropic":aniso, "inhomogenous_gaussian":varied}
for nm, vals in ds_dict.items():
    X, y = vals
    X = StandardScaler().fit_transform(X)

    plt.scatter(X[:, 0], X[:, 1], s=10, c=y)
    plt.savefig(outdir+"plot2D_" + str(n_samples) + "_" + nm + ".png")
    plt.close()

    # linear transformation
    X5_lin = np.matmul(X, A5)
    X20_lin = np.matmul(X, A20)

    # nonlinear transformation
    X_tensor = torch.tensor(X, dtype=torch.float32)
    X5_nonlin = MLP5(X_tensor).detach().numpy()
    X20_nonlin = MLP20(X_tensor).detach().numpy()

    np.savetxt(outdir + "y_" + str(n_samples) + "_" + nm + ".csv", y, delimiter=",")
    np.savetxt(outdir + "X_"+ str(n_samples) + "_" + nm + "2D.csv", X, delimiter=",")
    np.savetxt(outdir + "X_"+ str(n_samples) + "_" + nm + "5Dlin.csv", X5_lin, delimiter=",")
    np.savetxt(outdir + "X_"+ str(n_samples) + "_" + nm + "20Dlin.csv", X20_lin, delimiter=",")
    np.savetxt(outdir + "X_"+ str(n_samples) + "_" + nm + "5Dnonlin.csv", X5_nonlin, delimiter=",")
    np.savetxt(outdir + "X_"+ str(n_samples) + "_" + nm + "20Dnonlin.csv", X20_nonlin, delimiter=",")



