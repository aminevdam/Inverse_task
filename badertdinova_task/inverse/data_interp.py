import numpy as np
import scipy
from inverse_config import Pk, Rk, t_end, rc


def well_pressure():
    """well presure from experiment"""

    p_fd = np.genfromtxt('badertdinova_task/data/p_fd.csv', delimiter=',')

    t_fd = np.linspace(0, 1., len(p_fd))

    f = scipy.interpolate.interp1d(t_fd, p_fd)

    return f
