import numpy as np
import random
from numpy import linalg as LA


def F_r(S):
    if not S:
        F_val = 0
    else:
        F_val = max(S) - min(S) + 1
    return F_val


def L_loss(z, A, y):
    num_samples = A.shape[0]
    loss = 0.5 * (np.transpose(A@z - y) @ (A@z - y))[0][0] / (1.0 * num_samples)
    return loss


def G_ell(A, y, S):
    d = A.shape[1]
    # take subset S (feature)
    A_S = A[:, S]
    
    # solve w
    AtA = np.transpose(A_S) @ A_S
    Aty = np.transpose(A_S) @ y
    w = np.linalg.lstsq(AtA, Aty, rcond=None)[0]

    # update x
    x_iter = np.zeros((d, 1))
    x_iter[S] = w
    x_zeros = np.zeros((d, 1))
    
    # compute loss for G
    Ge_loss = L_loss(x_zeros, A, y) - L_loss(x_iter, A, y)

    return Ge_loss


def compute_subgrad(x, A, y, beta_reg):
    dim_x = x.shape[0]
    
    # sorting
    sort_index = np.argsort(x, axis=0)[::-1]
    Pi_list = np.transpose(sort_index).tolist()[0]

    F_vals = np.zeros((dim_x, 1))
    G_vals = np.zeros((dim_x, 1))

    F_vals[0] = F_r([Pi_list[0]])
    G_vals[0] = G_ell(A, y, [Pi_list[0]])
    
    F_grads = np.zeros((dim_x, 1))
    G_grads = np.zeros((dim_x, 1))
    
    F_grads[Pi_list[0]] = F_vals[0]
    G_grads[Pi_list[0]] = G_vals[0]

    for idx in range(1, dim_x):
        F_vals[idx] = F_r(Pi_list[:idx+1])
        G_vals[idx] = G_ell(A, y, Pi_list[:idx+1])
        F_grads[Pi_list[idx]] = F_vals[idx] - F_vals[idx-1]
        G_grads[Pi_list[idx]] = G_vals[idx] - G_vals[idx-1]

    H_vals = beta_reg * F_vals - G_vals
        
    return F_grads, G_grads, H_vals


def sample_S_t(x):
    dim_x = x.shape[0]
    
    # sorting
    sort_index = np.argsort(x, axis=0)[::-1]
    Pi_list = np.transpose(sort_index).tolist()[0]
    
    # compute lmbd
    lmbd_0 = 1.0
    lmbd_n_plus_1 = 0.0
    lmbd_vec = np.zeros((dim_x + 1, 1))
    lmbd_vec[0] = lmbd_0 - x[Pi_list[0]]
    lmbd_vec[-1] = x[Pi_list[-1]] - lmbd_n_plus_1
    for idx in range(1, dim_x):
        lmbd_vec[idx] = x[Pi_list[idx-1]] - x[Pi_list[idx]]
    
    # sampling
    S_t_index = np.random.choice(dim_x+1, 1, p=lmbd_vec.transpose()[0].tolist())[0]
    S_t = Pi_list[:S_t_index]
    if S_t_index == 0:
        S_t = []
    return S_t, S_t_index


