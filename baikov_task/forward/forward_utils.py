import torch
from forward_config import Pk, L

def mu_oil(grad):
    mu_h = 13.6e-3
    mu_o = 4.62e-3
    glad = 0.8e-3
    G = 0.25e+5
    mu = (mu_h-mu_o) * torch.special.expit (-Pk/L*glad * (abs(grad) - L*G/Pk)) + mu_o

    return mu