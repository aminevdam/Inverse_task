import torch
from forward_config import Pk, r0

def mu_oil(grad, u):
    mu_l = 25e-3
    mu_h = 10*mu_l
    B = 2e-3
    G = 0.005e+6
    mu = (mu_h - mu_l) * torch.special.expit(-Pk/r0 * B * (1/torch.exp(u)*grad - r0/Pk*G)) + mu_l

    return mu