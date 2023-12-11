import numpy as np
import scipy
import torch
from reference import finite_diffs
from forward_config import Pk, Rk, t_end


def fd_solution(grid):
    """Interpolate finite difference solution to another grid"""
    P, u, t = finite_diffs(300, 200)

    P = (P / Pk).T.reshape(-1)

    r = np.exp(u) / Rk

    t = t / t_end

    grid_fd = torch.cartesian_prod(torch.from_numpy(t), torch.from_numpy(r))

    u = scipy.interpolate.griddata(grid_fd, P, grid, method='nearest')

    return u
