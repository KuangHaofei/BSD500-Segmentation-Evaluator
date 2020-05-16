import sys
import numpy as np
from skimage import measure


def segmentation_cover(pred, gt):
    regionsGT = []
    regionsPred = []
    total_gt = 0

    cntR = 0
    sumR = 0
    cntP = 0
    sumP = 0

    propsGT = measure.regionprops(gt)
    for prop in propsGT:
        regionsGT.append(prop.area)
    regionsGT = np.array(regionsGT).reshape(-1, 1)
    total_gt = total_gt + np.max(gt)

    best_matchesGT = np.zeros((1, total_gt))

    matches = match_segmentation(pred, gt)

    matchesPred = np.max(matches, axis=1).reshape(-1, 1)
    matchesGT = np.max(matches, axis=0).reshape(1, -1)

    propsPred = measure.regionprops(pred)
    for prop in propsPred:
        regionsPred.append(prop.area)
    regionsPred = np.array(regionsPred).reshape(-1, 1)

    for r in range(regionsPred.shape[0]):
        cntP += regionsPred[r] * matchesPred[r]
        sumP += regionsPred[r]

    for r in range(regionsGT.shape[0]):
        cntR += regionsGT[r] * matchesGT[:, r]
        sumR += regionsGT[r]

    best_matchesGT = np.maximum(best_matchesGT, matchesGT)

    R = cntR / (sumR + (sumR == 0))
    P = cntP / (sumP + (sumP == 0))

    return R[0], P[0]


def match_segmentation(pred, gt):
    total_gt = np.max(gt)
    cnt = 0
    matches = np.zeros((total_gt, np.max(pred)))

    num1 = np.max(gt) + 1
    num2 = np.max(pred) + 1
    confcounts = np.zeros((num1, num2))

    # joint histogram
    sumim = 1 + gt + pred * num1

    hs, _ = np.histogram(sumim.flatten(), bins=np.linspace(1, num1*num2+1, num=num1*num2+1))
    hs = hs.reshape(confcounts.shape[1], confcounts.shape[0]).T

    confcounts = confcounts + hs
    accuracies = np.zeros((num1, num2))

    for j in range(0, num1):
        for i in range(0, num2):
            gtj = np.sum(confcounts[j, :])
            resj = np.sum(confcounts[:, i])
            gtjresj = confcounts[j, i]
            if gtj + resj - gtjresj:
                value = gtjresj / (gtj + resj - gtjresj)
            else:
                value = np.nan
            accuracies[j, i] = value
    matches[cnt:cnt + np.max(gt), :] = accuracies[1:, 1:]

    return matches.T
