import torch
from inverse_config import device, r0, betta, t_end,  Q, Pk, S
from inverse_utils import mu_oil


class DNN(torch.nn.Module):
    def __init__(self, model, G, k, C_skv, mu_h, tol, n_t):
        super(DNN, self).__init__()
        self.net = model
        self.tol = tol

        self.G = torch.tensor([G], requires_grad=False).float().to(device)
        self.G = torch.nn.Parameter(self.G)
        self.net.register_parameter('G', self.G)

        self.k = torch.tensor([k], requires_grad=False).float().to(device)
        self.k = torch.nn.Parameter(self.k)
        self.net.register_parameter('k', self.k)

        self.C_skv = torch.tensor([C_skv], requires_grad=False).float().to(device)
        self.C_skv = torch.nn.Parameter(self.C_skv)
        self.net.register_parameter('C_skv', self.C_skv)

        self.mu_h = torch.tensor([mu_h], requires_grad=False).float().to(device)
        self.mu_h = torch.nn.Parameter(self.mu_h)
        self.net.register_parameter('mu_h', self.mu_h)

        self.n_t = n_t
        self.tol = tol
        self.M = torch.triu(torch.ones((self.n_t, self.n_t)), diagonal=1).T.to(device)


    # Forward Feed
    def forward(self, x):
        return self.net(x)


    # Loss function for PDE
    def loss_pde(self, x):
        x.requires_grad=True
        y = self.net(x)
        p = y[:, 0:1].sum(0)

        dp_g, = torch.autograd.grad(p, x, create_graph=True)
        p_x, p_t = dp_g[:, 1:], dp_g[:, 0:1]

        p1 = 1./mu_oil(p_x, x[:,1:], self.G, self.mu_h)*p_x

        d2p_g, = torch.autograd.grad(p1.sum(0), x, create_graph=True)

        p_xx, p_xt = d2p_g[:, 1:], d2p_g[:, 0:1]

        # Loss function for the Euler Equations
        f = torch.exp(2*x[:,1:])*r0**2*betta/t_end/self.k*1e-14*p_t - p_xx

        return f

    def loss_weights(self, x):
        op = self.loss_pde(x).reshape(self.n_t, -1)
        L_t = torch.mean(op**2, axis=1).reshape(self.n_t, 1)
        with torch.no_grad():
            W = torch.exp(- self.tol * (self.M @ L_t))
        return L_t, W

    # Loss function for initial condition
    def loss_dirichlet(self, x_bc, p_bc):
        p_bc_nn = self.net(x_bc)[:, 0]

        # Loss function for the initial condition
        loss_bcs = ((p_bc_nn- p_bc) ** 2)

        return loss_bcs

    def loss_operator(self, x_bc):
        x_bc.requires_grad=True
        y = self.net(x_bc)
        p = y[:, 0:1].sum(0)
        dp_g, = torch.autograd.grad(p, x_bc, create_graph=True)
        p_x, p_t = dp_g[:, 1:], dp_g[:, 0:1]
        loss_bc = (p_x - Q * mu_oil(p_x, x_bc[:,1:], self.G, self.mu_h) / 
                   (Pk * self.k*1e-14 * S) - self.C_skv*1e-6 * mu_oil(p_x, x_bc[:,1:], self.G, self.mu_h) / (t_end * S * self.k*1e-14) * p_t)**2

        return loss_bc
