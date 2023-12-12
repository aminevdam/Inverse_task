import torch
from forward_utils import mu_oil
from forward_config import t_end, k, betta, L, Q, S, Pk, C_skv


class DNN(torch.nn.Module):
    def __init__(self, model):
        super(DNN, self).__init__()
        self.net = model
        self.loss_function = torch.nn.MSELoss(reduction ='mean')
        self.i = 0

    # Forward Feed
    def forward(self, x):
        return self.net(x)


    # Loss function for PDE
    def loss_pde(self, x):
        x.requires_grad = True
        y = self.net(x)
        p = y[:, 0:1].sum(0)

        dp_g, = torch.autograd.grad(p, x, create_graph=True)
        p_x, p_t = dp_g[:, :1], dp_g[:, 1:]

        p1 = p_x/mu_oil(p_x)

        d2p_g, = torch.autograd.grad(p1.sum(0), x, create_graph=True)

        p_xx, _ = d2p_g[:, :1], d2p_g[:, 1:]

        # Loss function for the Euler Equations
        f = p_t - t_end*k/(betta*L**2)*p_xx

        return torch.mean(f**2)

    # Loss function for initial condition
    def loss_dirichlet(self, x_bc, p_bc):
        p_bc_nn = self.net(x_bc)[:, 0]

        # Loss function for the initial condition
        loss_bcs = ((p_bc_nn - p_bc) ** 2)

        return loss_bcs

    def loss_operator(self, x_bc):
        x_bc.requires_grad=True
        y = self.net(x_bc)
        p = y[:, 0:1].sum(0)
        dp_g, = torch.autograd.grad(p, x_bc, create_graph=True)
        p_x, p_t = dp_g[:, 0], dp_g[:, 1]
        loss_bc = (p_x - mu_oil(p_x)*Q*L/(S*Pk*k) - C_skv*L/(t_end*S*k)*p_t*mu_oil(p_x))**2

        return loss_bc