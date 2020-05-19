import numpy as np
import sys


def match_segmentation_2(pred, gt):
    (tx, ty) = pred.shape

    num1 = np.max(pred)
    num2 = np.max(gt)
    confcounts = np.zeros((int(num1)+1, num2+1))

    for i in range(tx):
        for j in range(ty):
            u = pred[i, j]
            v = gt[i, j]

            confcounts[u, v] = confcounts[u, v] + 1

    RI = rand_index(confcounts)
    VI = variation_of_information(confcounts)

    return RI, VI


def rand_index(n):
    N = np.sum(n)
    n_u = np.sum(n, axis=1)
    n_v = np.sum(n, axis=0)

    N_choose_2 = N * (N - 1) / 2

    ri = 1 - (np.sum(n_u * n_u) / 2 + np.sum(n_v * n_v) / 2 - np.sum(n * n)) / N_choose_2

    return ri


def variation_of_information(n):
    N = np.sum(n)

    joint = n / N

    marginal_2 = np.sum(joint, axis=0)
    marginal_1 = np.sum(joint, axis=1)

    H1 = - np.sum(marginal_1 * np.log2(marginal_1 + (marginal_1 == 0)))
    H2 = - np.sum(marginal_2 * np.log2(marginal_2 + (marginal_2 == 0)))

    MI = np.sum(joint * log2_quotient(joint, np.dot(marginal_1.reshape(-1, 1), marginal_2.reshape(1, -1))))

    vi = H1 + H2 - 2 * MI

    return vi


def log2_quotient(A, B):
    lq = np.log2((A + ((A == 0) * B) + (B == 0)) / (B + (B == 0)))

    return lq
