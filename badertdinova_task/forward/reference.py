import numpy as np
import matplotlib.pyplot as plt
from forward_config import rc, Rk, t_end, Pk, S, k, C_skv, Q, betta

def finite_diffs(Nh, Nt):
    """finite difference solution"""

    A, B, C, F = np.zeros(Nh), np.zeros(Nh), np.zeros(Nh), np.zeros(Nh)
    alfa, beta = np.zeros(Nh), np.zeros(Nh)
    P, Ps, Pn = np.zeros((Nh,Nt)), np.zeros(Nh), np.zeros(Nh)
    grad = np.zeros(Nh)

    u = np.linspace(np.log(rc), np.log(Rk), Nh)
    h = (np.log(Rk)-np.log(rc)) / (Nh-1)
    t = np.linspace(0, t_end, Nt)
    tau = t[1]-t[0]

    epsilon = 1.e-5

    P[:,0] = Pk

    def mu_oil(grad, ux):
        mu_o = 25e-3
        mu_h = 10*mu_o
        B = 2e-3
        G = 0.005e+6
        return (mu_h-mu_o) / (1 + np.exp(B * (1/np.exp(ux)*grad - G))) + mu_o

    for j in range(1,Nt):
        Pn = P[:,j-1].copy()

        y = 0
        while True:
            if y==0:
                Ps = P[:,j-1].copy()
            else:
                Ps = P[:,j].copy()
            y += 1

            for i in range(1,Nh-1):
                grad[i] = (Ps[i+1]-Ps[i-1])/(2*h)
            grad[0] = (Ps[1]-Ps[0])/(h)
            grad[-1] = (Ps[-1]-Ps[-2])/(h)

            alfa[0] = (S*k/(mu_oil(grad[0], u[0])*h))/(C_skv/tau + S*k/(mu_oil(grad[0], u[0])*h))
            beta[0] = (-Q + C_skv/tau*Pn[0])/ (C_skv/tau + S*k/(mu_oil(grad[0], u[0])*h))


            for i in range(1, Nh-1):
                A[i] = 1. / h**2 * k / mu_oil(grad[i], u[i])
                B[i] = 1. / h**2 * k / mu_oil(grad[i], u[i])
                C[i] = A[i] + B[i] + betta*np.exp(2*u[i])/tau
                F[i] = betta*np.exp(2*u[i])/tau*Pn[i]

            for i in range(1, Nh-1):
                alfa[i]=A[i]/(C[i]-B[i]*alfa[i-1])
                beta[i]=(B[i]*beta[i-1]+F[i])/(C[i]-B[i]*alfa[i-1])

            P[-1][j] = Pk

            for i in range(Nh-2, -1, -1):
                P[i][j]=alfa[i]*P[i+1][j]+beta[i]

            error = abs((P[1][j]-Ps[1])/P[1][j])
            for i in range(Nh):
                if error < abs((P[i][j]-Ps[i])/P[i][j]):
                    error = abs((P[i][j]-Ps[i])/P[i][j])

            if error < epsilon:
                break
    return P, u, t

# P, u, t = finite_diffs(200, 20)

# plt.plot(np.exp(u), P[:,0]/Pk)
# plt.plot(np.exp(u), P[:,1]/Pk)
# plt.plot(np.exp(u), P[:,5]/Pk)
# plt.plot(np.exp(u), P[:,-1]/Pk)
# plt.show()
# plt.plot(t, P[0,:]/Pk)
# plt.xscale('log')
# plt.show()