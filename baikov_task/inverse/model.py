import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from inverse_utils import mu_oil
from inverse_config import t_end, k, betta, L, Q, S, Pk, C_skv, device

velocity = np.loadtxt("data/W.txt", delimiter='\t', dtype=np.float64)
gradient = np.loadtxt("data/grad.txt", delimiter='\t', dtype=np.float64)

velocity = velocity[:-3]*1e7  # W*1e+7 m/s*1e+7
gradient = gradient[:-3]/1e5 # atm/m

velocity[0] = 0
gradient[0] = 0

f = interpolate.interp1d(gradient, velocity)

class DNN(torch.nn.Module):
    def __init__(self, model, G, glad, mu_h):
        super(DNN, self).__init__()
        self.net = model
        self.loss_function = torch.nn.MSELoss(reduction ='mean')
        'Initialize our new parameters i.e. G (Inverse problem)'
        self.G = torch.tensor([G], requires_grad=True).float().to(device)
        self.glad = torch.tensor([glad], requires_grad=True).float().to(device)
        self.mu_h = torch.tensor([mu_h], requires_grad=True).float().to(device)
        'Register G to optimize'
        self.G = torch.nn.Parameter(self.G)
        self.glad = torch.nn.Parameter(self.glad)
        self.mu_h = torch.nn.Parameter(self.mu_h)
        'Register our new parameter'
        self.net.register_parameter('G', self.G)
        self.net.register_parameter('glad', self.glad)
        self.net.register_parameter('mu_h', self.mu_h)
        self.i = 0
        self.velocity = []
        self.dp_grad = []
    # Forward Feed
    def forward(self, x):
        return self.net(x)

    def prepare_idx(self, dp_dx, grad):
        dp_dx = dp_dx*Pk/L/1e5
        dp_dx1 = dp_dx[dp_dx>=0.]
        dp_dx1 = dp_dx1[dp_dx1<=0.62]
        idx = []
        for point in dp_dx1:
            index = np.where(dp_dx==point)[0][0]
            idx.append(int(index))
            #dp_dx1 = np.delete(dp_dx1, np.where(dp_dx1==point)[0])
        return idx


    def loss_data(self, x, grad, data):
        x.requires_grad = True
        y = self.net(x)
        p = y[:, 0:1].sum(0)

        dp_g, = torch.autograd.grad(p, x, create_graph=True)

        p_x = dp_g[:, 0]

        dp_dx = p_x.detach().cpu().numpy()

        idx = self.prepare_idx(dp_dx, grad)

        w = k*p_x[idx]/mu_oil(p_x[idx], self.G, self.glad, self.mu_h)*Pk/L*1e7
        w1 = k*p_x/mu_oil(p_x, self.G, self.glad, self.mu_h)*Pk/L*1e7
        w_data = torch.from_numpy(data(p_x[idx].detach().cpu().numpy()*Pk/L/1e5)).float().to(device)
        if self.i%2000==0:
            plt.plot(dp_dx*Pk/L/1e5, w1.detach().cpu().numpy(), '+')
            plt.plot(gradient, f(gradient), '-.', label = 'Интерполяция')
            plt.ylim(0, 0.8)
            plt.xlim(0, 1.5)
            plt.show()
        if self.i%100==0:
            self.velocity.append(w.detach().cpu().numpy())
            self.dp_grad.append(dp_dx[idx]*Pk/L/1e5)
        self.i += 1
        loss_data = (w - w_data)**2

        if loss_data.shape[0] == 0:
            return 0.
        else:
            return loss_data

    # Loss function for PDE
    def loss_pde(self, x):
        x.requires_grad = True
        y = self.net(x)
        p = y[:, 0:1].sum(0)

        dp_g, = torch.autograd.grad(p, x, create_graph=True)
        p_x, p_t = dp_g[:, :1], dp_g[:, 1:]

        p1 = p_x/mu_oil(p_x, self.G, self.glad, self.mu_h)

        d2p_g, = torch.autograd.grad(p1.sum(0), x, create_graph=True)

        p_xx, p_xt = d2p_g[:, :1], d2p_g[:, 1:]

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
        loss_bc = (p_x - mu_oil(p_x, self.G, self.glad, self.mu_h)*Q*L/(S*Pk*k) - C_skv*L/(t_end*S*k)*p_t*mu_oil(p_x, self.G, self.glad, self.mu_h))**2

        return loss_bc
