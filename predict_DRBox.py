'''
Predict bounding boxes usiing a DRBox model
You can either plot the predictions or save them a jpg images

For example it can be run with the following command :

python predict_DRBox.py -i data/Car/train_data/ -m trained_models/car.h5 -l data/Car/labelstest.csv -c 0.4

Copyright © 2019 THALES ALENIA SPACE FRANCE. All rights reserved

Author : Paul Pontisso
'''

from keras.models import load_model

from keras_loss_function.keras_drbox_loss import DRBoxLoss
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization

from drbox_encoder_decoder.drbox_output_decoder import decode_detections
from data_generator.object_detection_2d_data_generator import DataGenerator

from tqdm import tqdm
from bounding_box_utils.visualization import visualize
import argparse
import os
import warnings
import numpy as np

# Deactivate the irrelevant warnings likely to occur
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=np.RankWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Lower the verbosity of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Command line argument parsing
parser = argparse.ArgumentParser(description='Script to predict bounding boxes around objects with DRBox model')

parser.add_argument('-i', '--images_folder',
                    type=str,
                    required=True,
                    help="Data folder to read the image and labels.")

parser.add_argument('-m', '--model_name',
                    type=str,
                    required=True,
                    help="The h5 file containing the model")

parser.add_argument('-l', '--labels',
                    type =str,
                    default=None,
                    help="CSV file containing the test set labels")

parser.add_argument('-f', '--save_folder',
                    type=str,
                    default=None,
                    help="Folder to save the images with predicted boxes if set")

parser.add_argument('-c', '--confidence',
                    type=float,
                    default=0.1,
                    help="The confidence threshold to apply before visualizing")

args = parser.parse_args()

#                          COMMAND LINE PARAMETERS
# ______________________________________________________________________________

confidence = args.confidence
images_folder = args.images_folder
labels = args.labels
model_name = args.model_name
save_folder = args.save_folder

normalize_coords = True
img_height = 300  # Height of the model input images
img_width = 300  # Width of the model input images
n_classes = 1

# Set the generator for the predictions.
val_dataset = DataGenerator()

if labels:
    val_dataset = DataGenerator()
    val_dataset.parse_csv(images_dir=images_folder,
                          labels_filename=labels,
                          include_classes='all')
    ret = {'processed_images', 'original_labels', 'filenames'}
else:
    filenames = os.listdir(images_folder)
    val_dataset = DataGenerator(filenames = filenames)
    ret = {'processed_images', 'filenames'}

predict_generator = val_dataset.generate(batch_size=1,
                                         shuffle=True,
                                         transformations=[],
                                         label_encoder=None,
                                         returns=ret,
                                         keep_images_without_gt=False)

val_dataset_size = val_dataset.get_dataset_size()
print('dataset size : ', val_dataset_size)

# load model
drbox_loss = DRBoxLoss(neg_pos_ratio=3, alpha=1.0)

# Set the path to the model you want to evaluate
model_drbox = load_model(model_name, custom_objects={'L2Normalization': L2Normalization, 'AnchorBoxes': AnchorBoxes,
                                                     'compute_loss': drbox_loss.compute_loss})


# Predict the boxes for every image in the validation dataset

print('predictions')

for i in tqdm(range(val_dataset_size)):

    if labels:
        batch_images, batch_filenames, batch_original_labels = next(predict_generator)
    else:
        batch_images, batch_filenames = next(predict_generator)
        batch_original_labels = [None]

    # predict boxes with the DRBox model
    y_pred = model_drbox.predict(batch_images)

    y_pred_decoded = decode_detections(y_pred,
                                       confidence_thresh=confidence,
                                       iou_threshold=0.01,
                                       top_k=100,
                                       normalize_coords=normalize_coords,
                                       img_height=img_height,
                                       img_width=img_width)

    if save_folder is None:
        visualize(batch_images[0], gt_labels=batch_original_labels[0], pred_labels=y_pred_decoded[0])

    else:

        save_path = os.path.join(save_folder, batch_filenames[0])

        visualize(batch_images[0], gt_labels=batch_original_labels[0], pred_labels=y_pred_decoded[0],
                  save_path=save_path)
