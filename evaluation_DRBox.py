'''
Evaluate DRBox models by plotting roc curve and calculating Average Precision

For example it can be run with the following command :

python evaluation_DRBox.py -i ../data/Car/train_data/ -m trained_models/car.h5 -l ../data/Car/labelstest.csv -n 605

Copyright © 2019 THALES ALENIA SPACE FRANCE. All rights reserved

Author : Paul Pontisso
'''

import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model

from keras_loss_function.keras_drbox_loss import DRBoxLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization

from drbox_encoder_decoder.drbox_output_decoder import decode_detections
from data_generator.object_detection_2d_data_generator import DataGenerator

from tqdm import tqdm
from bounding_box_utils.bounding_box_utils import iou
import warnings
import os
import argparse

# Deactivate the irrelevant warnings likely to occur
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=np.RankWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Lower the verbosity of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


# Command line argument parsing
parser = argparse.ArgumentParser(description='Script to evaluate a DRBox model by calculating roc curve and average precision')

parser.add_argument('-i', '--images_folder',
                    type=str,
                    required=True,
                    help="Data folder to read the image and labels.")

parser.add_argument('-m', '--model_name',
                    type=str,
                    required=True,
                    help="Path to the h5 file containing the model")

parser.add_argument('-l', '--labels',
                    type =str,
                    default=None,
                    help="Whether or not to shuffle the images")

parser.add_argument('-n', '--number',
                    type =int,
                    default=600,
                    help="number of test example to use for the evaluation")

args = parser.parse_args()

#                          COMMAND LINE PARAMETERS
# ______________________________________________________________________________

images_folder = args.images_folder
labels = args.labels
model_name = args.model_name
nb_test_images = args.number


normalize_coords = True
img_height = 300  # Height of the model input images
img_width = 300  # Width of the model input images
n_classes = 1

# Set the generator for the predictions.
val_dataset = DataGenerator()

val_dataset.parse_csv(images_dir=images_folder,
                      labels_filename=labels,
                      include_classes='all')

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=False,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)

val_dataset_size = val_dataset.get_dataset_size()

# number of test set images to use during evaluation

# number of bound not cut

num_objects = 0
for i in range(nb_test_images):
    batch_images, batch_original_labels = next(predict_generator)
    num_objects += len(batch_original_labels[0][batch_original_labels[0][:, 0] == 1])

# load model
drbox_loss = DRBoxLoss(neg_pos_ratio=3, alpha=1.0)

# Set the path to the model you want to evaluate
model_drbox = load_model(model_name, custom_objects={'L2Normalization': L2Normalization, 'AnchorBoxes': AnchorBoxes,
                                                     'compute_loss': drbox_loss.compute_loss})

# ----------------------------------------------------------------------------------------------------------------------
# Draw ROC curve by changing the confidence threshold
# ----------------------------------------------------------------------------------------------------------------------

# Predict the boxes for every image in the validation dataset befor drawing the roc curve 
# so that we don't have to calculate this every time and save time

print('Predictions')
predictions = []

val_dataset = DataGenerator()

val_dataset.parse_csv(images_dir=images_folder,
                      labels_filename=labels,
                      include_classes='all')

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=False,
                                         transformations=[],
                                         label_encoder=None,
                                         returns={'processed_images',
                                                  'original_labels'},
                                         keep_images_without_gt=False)

for i in tqdm(range(nb_test_images)):

    batch_images, batch_original_labels = next(predict_generator)

    # predict boxes with the DRBox model
    y_pred = model_drbox.predict(batch_images)
    predictions.append(y_pred)

predictions = np.array(predictions)

# Set the different iou_threshold you want to plot the ROC curve
for iou_threshold in [0.1, 0.5]:
    roc = []
    precision = []
    recall = []

    # we evaluate the model with different confidence threshold, from 0.1 to 0.9
    for conf in tqdm([x / 20.0 for x in range(2, 20)]):

        false_positive = 0
        false_negative = 0
        true_positive = 0

        val_dataset = DataGenerator()

        val_dataset.parse_csv(images_dir=images_folder,
                              labels_filename=labels,
                              include_classes='all')

        # Set the generator for the predictions.
        predict_generator = val_dataset.generate(batch_size=1,
                                                 shuffle=False,
                                                 transformations=[],
                                                 label_encoder=None,
                                                 returns={'processed_images',
                                                          'original_labels'},
                                                 keep_images_without_gt=False)

        # for every image in the validation dataset
        for j in tqdm(range(nb_test_images)):

            batch_images, batch_original_labels = next(predict_generator)
            i = 0  # Which batch item to look at

            # get sample from previously calculated predictions
            y_pred = predictions[j]

            # Decode the raw predictions in `y_pred`.

            y_pred_decoded = decode_detections(y_pred,
                                               confidence_thresh=conf,
                                               iou_threshold=0.35,
                                               top_k=100,
                                               normalize_coords=normalize_coords,
                                               img_height=img_height,
                                               img_width=img_width)

            # calulate the number of false positive and true positive in the image

            # get the ground truth labels
            gt_labels = np.copy(batch_original_labels[i])
            # for every box predicted by the model
            for boxp in y_pred_decoded[i]:
                # if the box predicted a bond not cut
                if boxp[0] == 1:
                    # We consider the predicted box as not matched until we can prove it is
                    matched = False

                    max_iou = 0
                    ind = -1

                    # for every ground truth box
                    for row, boxgt in enumerate(gt_labels):
                        # if the bond is not cut
                        if boxgt[0] == 1:

                            # calculate the iou between the predicted box and the ground truth box
                            inter_over_union = iou(boxp[2:], boxgt[1:])
                            # if the boxes are close enough we consider the prediction as a good prediction => a true positive
                            if inter_over_union > max_iou:

                                max_iou = inter_over_union
                                ind = row

                    # if one of the boxes is close enough we consider the prediction as a good prediction => a true positive
                    if max_iou > iou_threshold:

                        true_positive += 1

                        # we remove the ground truth box so that it is not matched twice
                        gt_labels = np.delete(gt_labels, ind, 0)

                        # we now consider the predicted box as matched
                        matched = True

                    # In the case the predicted box has no ground truth box with an iou over the threshold
                    # then we consider this box as a false positive
                    if not matched:
                        false_positive += 1

        # This is a new point in the roc curve
        roc.append([true_positive / num_objects, false_positive / num_objects])
        precision.append(true_positive / (true_positive + false_positive))
        recall.append(true_positive / num_objects)

    # compute average precision
    precision = np.array(precision)
    recall = np.array(recall)

    prec_at_rec = []

    for recall_level in np.linspace(0.0, 1.0, 1000):
        try:
            args = np.argwhere(recall >= recall_level).flatten()
            prec = max(precision[args])
        except ValueError:
            prec = 0.0

        prec_at_rec.append(prec)
    avg_prec = np.mean(prec_at_rec)

    print('Average precision of model for iou of {} : {}'.format(iou_threshold, avg_prec))

    roc = np.array(roc)
    print(roc)
    # plot the roc curve associated with one iou_threshold
    plt.plot(roc[:, 1], roc[:, 0], label='iou_threshold = ' + str(iou_threshold))
    plt.legend()
    plt.xlabel('False positive rate')
    plt.ylabel('True positive rate')
    plt.grid(color='grey', linestyle='--', linewidth=1)

plt.show()
# save figure
plt.savefig('courbe roc.jpg')
