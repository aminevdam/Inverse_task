import torch
from inverse_config import Pk, r0


def mu_oil(grad, u, G, mu_h):
    mu_l = 25e-3
    B = 2e-3
    mu = (mu_h*mu_l - mu_l) * torch.special.expit(
        -Pk / r0 * B * (1/torch.exp(u) * grad - r0 / Pk * G*1e+4)) + mu_l

    return mu