import torch
from inverse_config import Pk, L

def mu_oil(grad, G, glad, mu_h):
    mu_o = 5.75e-3
    mu = (mu_h*1e-3 - mu_o) * torch.special.expit (-Pk/L*glad*1e-3 * (abs(grad) - G)) + mu_o  #L*G/Pk

    return mu