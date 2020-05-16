import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec

from dataloader import DataLoader
from match_segmentation import segmentation_cover
from match_segmentation_2 import match_segmentation_2

if __name__ == '__main__':

    test_loader = DataLoader()

    scores = 0
    sumRI = 0
    sumVI = 0

    datasize = len(test_loader)

    for idx, (raw_img, raw_gt, mean_gt, raw_pred, mean_pred) in enumerate(test_loader):
        print('Image ', idx)

        gt = raw_gt[0]
        pred = raw_pred[0]

        # visualization
        plt.figure(figsize=(20, 4))
        grid_spec = gridspec.GridSpec(1, 3, width_ratios=[3, 3, 3])

        plt.subplot(grid_spec[0])
        plt.axis('off')
        plt.title('RGB image')
        plt.imshow(raw_img)

        plt.subplot(grid_spec[1])
        plt.axis('off')
        plt.title('prediction')
        plt.imshow(pred)

        plt.subplot(grid_spec[2])
        plt.axis('off')
        plt.title('ground Truth')
        plt.imshow(gt)

        plt.show()

        # PRI and VI
        ri, vi = match_segmentation_2(pred, gt)

        sumRI += ri
        sumVI += vi

        # SC
        R, _ = segmentation_cover(pred, gt)

        scores += R

        print('SC is: ', R)
        print('RI is: ', ri)
        print('VI is: ', vi)

    SC = scores / datasize
    PRI = sumRI / datasize
    VI = sumVI / datasize

    print("==================================")
    print('mean SC is: ', SC)
    print('mean PRI is: ', PRI)
    print('mean VI is: ', VI)
