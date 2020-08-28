import pandas as pd
import numpy as np
import torch
import torch.nn as nn


class AEDR(nn.Module):
    def __init__(self, input_dim, n_enc_layers, n_dec_layers, n_units, noise=0.0, z0_dim=2, p_drop=0.3):
        """
        :param input_dim:
        :param n_hid_layers: number of hidden layers in encoder/decoder
        :param n_units: number of units in hidden layers of encoder/decoder
        :param z0_dim: dimension of low-dimensional representation
        :param p_drop:
        """
        super(AEDR, self).__init__()

        lyrs_enc = []
        for i in range(n_enc_layers):
            lyrs_enc.append(nn.Linear(n_units, n_units))
            lyrs_enc.append(nn.ReLU(True))  # True (for inplace) necessary or not?
            lyrs_enc.append(nn.Dropout(p_drop))
        lyrs_enc.append(nn.Linear(n_units, z0_dim))

        lyrs_dec = []
        for i in range(n_dec_layers):
            lyrs_dec.append(nn.Linear(n_units, n_units))
            lyrs_dec.append(nn.ReLU(True))
            lyrs_dec.append(nn.Dropout(p_drop))
        lyrs_dec.append(nn.Linear(n_units, input_dim))

        self.encoder = torch.nn.Sequential(*self.make_MLP(dim_in=input_dim, dim_out=z0_dim, n_layers=n_enc_layers,
                                                          n_units=n_units, p_drop=p_drop))
        self.decoder = torch.nn.Sequential(*self.make_MLP(dim_in=z0_dim, dim_out=input_dim, n_layers=n_dec_layers,
                                                          n_units=n_units, p_drop=p_drop))

        self.noise = noise

    def make_MLP(self, dim_in, dim_out, n_layers, n_units, p_drop=.3):
        lyrs = []
        if n_layers == 0:  # simple linear mapping
            lyrs.append(nn.Linear(dim_in, dim_out))
        else:
            lyrs.append(nn.Linear(dim_in, n_units))
            lyrs.append(nn.ReLU(True))
            lyrs.append(nn.Dropout(p_drop))
            for i in range(n_layers-1): # if 2+ hidden layers are to be used
                lyrs.append(nn.Linear(n_units, n_units))
                lyrs.append(nn.ReLU(True))
                lyrs.append(nn.Dropout(p_drop))
            lyrs.append(nn.Linear(n_units, dim_out))
        return lyrs



    def forward(self, X):
        z0 = self.encoder(X)
        # add noise to z0 to make a more robust representation
        z0_noisy = z0 + torch.normal(0, self.noise, size=z0.size())
        return z0, self.decoder(z0_noisy)

def gaussian_KL(mu_1, mu_2, sigma_1, sigma_2):
    return(torch.log(sigma_2) - torch.log(sigma_1) + (torch.pow(sigma_1,2)+torch.pow((mu_1 - mu_2),2)) / (2*sigma_2**2) - 0.5)

def l2loss(X):
    return torch.sum(torch.pow(X,2))

def l3loss(X):
    return torch.sum(torch.pow(torch.abs(X),3))

def matrix_L1(A, B):
    assert A.shape == B.shape, "matrix dimensions don't match"
    assert len(A.shape) == 2, "unexpected number of dimensions"
    assert A.shape[0]==A.shape[1], "unexpected dimensions"
    return np.sum(np.sum(np.abs(A - B), axis=-1), axis=-1)/(A.shape[0]**2)