import numpy as np
from forward_config import L, t_end, Pk, C_skv, S, k, Q, betta

def finite_diffs(Nh, Nt):

    A, B, C, F = np.zeros(Nh), np.zeros(Nh), np.zeros(Nh), np.zeros(Nh)
    alfa, beta = np.zeros(Nh), np.zeros(Nh)
    P, Ps, Pn = np.zeros((Nh,Nt)), np.zeros(Nh), np.zeros(Nh)
    grad = np.zeros(Nh)


    h = (L)/(Nh-1)
    x = np.linspace(0, L, Nh)


    tau = t_end / (Nt-1)
    t = np.linspace(0,t_end,Nt)

    epsilon = 1.e-5

    P[:,0] = Pk

    def mu_oil(grad):
        mu_h = 13.6e-3
        mu_o = 4.62e-3
        glad = 0.8e-3
        G = 0.25e+5
        return (mu_h-mu_o) / (1 + np.exp(glad * (grad - G))) + mu_o

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


            alfa[0] = (S*k/(mu_oil(grad[0])*h))/(C_skv/tau + S*k/(mu_oil(grad[0])*h))
            beta[0] = (-Q + C_skv/tau*Pn[0])/ (C_skv/tau + S*k/(mu_oil(grad[0])*h))


            for i in range(1, Nh-1):
                A[i] = 1. / h**2 * k / mu_oil(grad[i])
                B[i] = 1. / h**2 * k / mu_oil(grad[i])
                C[i] = A[i] + B[i] + betta/tau
                F[i] = betta/tau*Pn[i]

            for i in range(1, Nh-1):
                alfa[i]=A[i]/(C[i]-B[i]*alfa[i-1])
                beta[i]=(B[i]*beta[i-1]+F[i])/(C[i]-B[i]*alfa[i-1])

            P[-1][j] = Pk

            for i in range(Nh-2, -1, -1):
                P[i][j]=alfa[i]*P[i+1][j]+beta[i]

            u = abs((P[1][j]-Ps[1])/P[1][j])
            for i in range(Nh):
                if u < abs((P[i][j]-Ps[i])/P[i][j]):
                    u = abs((P[i][j]-Ps[i])/P[i][j])

            if u < epsilon:
                break

    return P, x, t
