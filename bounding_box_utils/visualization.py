'''
Visualization tool to help visualizing bounding boxes around objects
This can be used before training to make sure the inputs are ok, or during predictions

Copyright © 2019 THALES ALENIA SPACE FRANCE. All rights reserved

Author : Paul Pontisso
'''

import numpy as np
import matplotlib.pyplot as plt


def visualize(image, gt_labels=None, pred_labels=None, save_path=None):
    '''
    Visualization tool to plot images with ground truth boxes and/or predicted labels

    Arguments:
        image (numpy array) : the image to plot
        gt_labels (np array, optional) : the ground truth labels, this will be plot in red
        pred_labels (np array, optional) : the predicted labels, this will be plot in blue
        save_folder (string, optional) : if set, the image is not plot but saved with this path

    '''

    plt.figure(figsize=(12, 12))
    plt.imshow(image)
    current_axis = plt.gca()

    # draw ground truth boxes
    if gt_labels is not None:

        current_axis.text(1, -25, '- Ground Truth', size='x-large', color='blue', bbox={'facecolor': 'white', 'alpha': 1.0})

        for box in gt_labels:
            cx = box[1]
            cy = box[2]
            w = box[3]
            h = box[4]
            angle = box[5]

            xmin, ymin = cx - 1 / 2 * (-h * np.sin(angle) + w * np.cos(angle)), \
                         cy - 1 / 2 * (h * np.cos(angle) + w * np.sin(angle))

            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), w, h, angle=angle * 180 / np.pi, color='red', fill=False, linewidth=2))

    # draw predicted boxes
    if pred_labels is not None:

        current_axis.text(1, -10, '- Predicted boxes', size='x-large', color='red', bbox={'facecolor': 'white', 'alpha': 1.0})

        for box in pred_labels:
            cx = box[2]
            cy = box[3]
            w = box[4]
            h = box[5]
            angle = box[6]

            xmin, ymin = cx - 1 / 2 * (-h * np.sin(angle) + w * np.cos(angle)), \
                         cy - 1 / 2 * (h * np.cos(angle) + w * np.sin(angle))

            current_axis.add_patch(
                plt.Rectangle((xmin, ymin), w, h, angle=angle * 180 / np.pi, color='blue', fill=False, linewidth=2))

            label = '{:.2f}'.format(box[1])
            current_axis.text(xmin, ymin, label, size='x-small', color='white', bbox={'facecolor': 'blue', 'alpha': 1.0})

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()
