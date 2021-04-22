'''

DRBox Training
This script can be used to train a new model or continue the training of an old model

You must specify the training parameters you want to use, either airplanes, ships or vehicles
you must also specify the csv file containing the labels that you wish to use during training and validation

Copyright © 2019 THALES ALENIA SPACE FRANCE. All rights reserved

Author : Paul Pontisso
'''

from keras.optimizers import Adam, SGD
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, TerminateOnNaN, CSVLogger
from keras import backend as K
from math import ceil
import numpy as np
import os
import warnings

from models.keras_DRBox import drbox
from keras_loss_function.keras_drbox_loss import DRBoxLoss

from keras.models import load_model
from keras_layers.keras_layer_AnchorBoxes import AnchorBoxes
from keras_layers.keras_layer_L2Normalization import L2Normalization

from drbox_encoder_decoder.drbox_input_encoder import DRBoxInputEncoder

from data_generator.object_detection_2d_data_generator import DataGenerator
from data_generator.data_augmentation_chain_drbox import DataAugmentation

# Deactivate the irrelevant warnings likely to occur
warnings.simplefilter(action='ignore', category=UserWarning)
warnings.simplefilter(action='ignore', category=np.RankWarning)
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=RuntimeWarning)

# Lower the verbosity of TensorFlow
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# 1. Set the model configuration parameters

img_height = 300  # Height of the model input images
img_width = 300  # Width of the model input images
img_channels = 3  # Number of color channels of the model input images

# The per-channel mean of the images in the dataset. Do not change this value if you're using any of the
# pre-trained weights.
mean_color = [123, 117, 104]
n_classes = 1  # Number of positive classes


##################################################################################
# Aiplanes Parameters
##################################################################################

scales = [0.1, 0.14, 0.17, 0.22, 0.27, 0.34]  # The anchor boxes scaling factors
aspect_ratios = [[1, 1.4],
                 [1, 1.4],
                 [1, 1.4],
                 [1, 1.4],
                 [1, 1.4],
                 [1, 1.4]]  # The anchor boxes aspect ratios

angles = [i * np.pi / 6 for i in range(6)]  # the anchor boxes angles

pos_ariou_threshold = 0.4
neg_ariou_limit = 0.2

# The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the
# step size for each predictor layer.
offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# The variances by which the encoded target coordinates are divided as in the original implementation os SSD
variances = [0.1, 0.1, 0.2, 0.2, 1]
# learning rate
lr = 0.0001
# l2 regularization parameter
l2_reg = 0
alpha = 1

# Set the paths to the datasets here.
images_dir = 'data/Airplane/train_data/'

train_labels_filename = "C:\\Users\\handenur.caliskan\\Documents\\GitHub\\DRBox_keras\\train_labels.csv"
val_labels_filename = "C:\\Users\handenur.caliskan\\Documents\\GitHub\\DRBox_keras\\validation_labels.csv"

proba_no_aug = 1/3

##################################################################################
# Ships Parameters
##################################################################################

# scales = [0.04, 0.08, 0.1, 0.13, 0.16, 0.2]  # The anchor boxes scaling factors
# aspect_ratios = [[ 2, 3, 4],
#                  [ 2, 3, 4],
#                  [ 2.5, 3.5],
#                  [ 2.5, 3.5],
#                  [ 2.5, 3.5],
#                  [ 2.5, 3.5]]  # The anchor boxes aspect ratios

# angles = [i * np.pi / 9 for i in range(9)]  # the anchor boxes angles

# pos_ariou_threshold = 0.4
# neg_ariou_limit = 0.3

# # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the
# # step size for each predictor layer.
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# # The variances by which the encoded target coordinates are divided as in the original implementation of SSD
# variances = [0.1, 0.1, 0.2, 0.2, 1]
# # learning rate
# lr = 0.0001
# # l2 regularization parameter
# l2_reg = 0
# alpha = 3.0

# # Set the paths to the datasets here.
# images_dir = 'C:\\Users\\handenur.caliskan\\Documents\\GitHub\\DRBox_keras\\training_kimlik'

# train_labels_filename = "C:\\Users\\handenur.caliskan\\Documents\\GitHub\\DRBox_keras\\train_labels.csv"
# val_labels_filename = "C:\\Users\handenur.caliskan\\Documents\\GitHub\\DRBox_keras\\validation_labels.csv"

# proba_no_aug = 0.5

##################################################################################
#  Vehicles Parameters
##################################################################################
#
# scales = [0.018, 0.018, 0.018, 0.018, 0.018, 0.018]  # The anchor boxes scaling factors
# aspect_ratios = [[ 2.7],
#                  [ 2.7],
#                  [ 2.7],
#                  [ 2.7],
#                  [ 2.7],
#                  [ 2.7]] # The anchor boxes aspect ratios
#
# angles = [i * np.pi / 6 for i in range(6)]  # the anchor boxes angles
# pos_ariou_threshold = 0.3
# neg_ariou_limit = 0.01
#
# # The offsets of the first anchor box center points from the top and left borders of the image as a fraction of the
# # step size for each predictor layer.
# offsets = [0.5, 0.5, 0.5, 0.5, 0.5, 0.5]
# # The variances by which the encoded target coordinates are divided as in the original implementation os SSD
# variances = [0.1, 0.1, 0.2, 0.2, 1]
# # learning rate
# lr = 0.0001
# # l2 regularization parameter
# l2_reg = 0.005
# alpha = 1
#
# # Set the paths to the datasets here.
# images_dir = 'data/Car/train_data/'
#
# train_labels_filename = "data/Car/labelstrain.csv"
# val_labels_filename = "data/Car/labelsvalidate.csv"
#
# proba_no_aug = 1/3

normalize_coords = True

# 2. Create DRBox model
# You can use either part 2.1 if you want to create model from scratch or 2.2 if you want to load a trained model
# but you should not use both at the same time
# 
# 2.1 Build the model from scratch

# Build the Keras model.

K.clear_session()  # Clear previous models from memory.

model = drbox(image_size=(img_height, img_width, img_channels),
              n_classes=n_classes,
              l2_regularization=l2_reg,
              scales=scales,
              aspect_ratios_per_layer=aspect_ratios,
              angles_global=angles,
              offsets=offsets,
              variances=variances,
              normalize_coords=normalize_coords,
              subtract_mean=mean_color)

# Load some weights into the model.
# uncomment if you want to load pretrained weights
# Set the path to the weights you want to load.

# weights_path = 'VGG_weights/VGG_ILSVRC_16_layers_fc_reduced.h5'
# if os.path.exists(weights_path):
#     model.load_weights(weights_path, by_name=True)

# Instantiate an Adam optimizer and the DRbox loss function and compile the model.

adam = Adam(lr=lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.0)

drbox_loss = DRBoxLoss(neg_pos_ratio=3, alpha=alpha)

model.compile(optimizer=adam, loss=drbox_loss.compute_loss)

# 2.2 Build the model from a previously trained model
# uncomment if you want to use a pre trained model

# drbox_loss = DRBoxLoss(neg_pos_ratio=3, alpha=1.0)
# model_name = 'trained_models/1model.h5'
# model = load_model(model_name, custom_objects={'L2Normalization': L2Normalization, 'AnchorBoxes': AnchorBoxes,
#                                                'compute_loss': drbox_loss.compute_loss})

# 3. Set up the data generators for the training

# Instantiate two `DataGenerator` objects: One for training, one for validation.

train_dataset = DataGenerator(load_images_into_memory=False, show_images=False)
val_dataset = DataGenerator(load_images_into_memory=False)

# Parse the image and label lists for the training and validation datasets.

train_dataset.parse_csv(images_dir=images_dir,
                        labels_filename=train_labels_filename,
                        include_classes='all')

val_dataset.parse_csv(images_dir=images_dir,
                      labels_filename=val_labels_filename,
                      include_classes='all')

# Set the batch size.

batch_size = 6

# Set the image transformations for pre-processing and data augmentation options.

# For the training generator:
data_augmentation = DataAugmentation(img_height, img_width, proba_no_aug=proba_no_aug)

# Instantiate an encoder that can encode ground truth labels into the format needed by the DRBox loss function.

# The encoder constructor needs the spatial dimensions of the model's predictor layers to create the anchor boxes.
predictor_sizes = [model.get_layer('conv4_3_norm_mbox_conf').output_shape[1:3],
                   model.get_layer('fc7_mbox_conf').output_shape[1:3],
                   model.get_layer('conv6_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv7_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv8_2_mbox_conf').output_shape[1:3],
                   model.get_layer('conv9_2_mbox_conf').output_shape[1:3]]

drbox_input_encoder = DRBoxInputEncoder(img_height=img_height,
                                        img_width=img_width,
                                        n_classes=n_classes,
                                        predictor_sizes=predictor_sizes,
                                        scales=scales,
                                        aspect_ratios_per_layer=aspect_ratios,
                                        angles_global=angles,
                                        offsets=offsets,
                                        variances=variances,
                                        matching_type='multi',
                                        pos_ariou_threshold=pos_ariou_threshold,
                                        neg_ariou_limit=neg_ariou_limit,
                                        normalize_coords=normalize_coords)

# Create the generator handles that will be passed to Keras' `fit_generator()` function.

train_generator = train_dataset.generate(batch_size=batch_size,
                                         shuffle=True,
                                         transformations=[data_augmentation],
                                         label_encoder=drbox_input_encoder,
                                         returns={'processed_images',
                                                  'encoded_labels'},
                                         keep_images_without_gt=False)

val_generator = val_dataset.generate(batch_size=batch_size,
                                     shuffle=True,
                                     transformations=[],
                                     label_encoder=drbox_input_encoder,
                                     returns={'processed_images',
                                              'encoded_labels'},
                                     keep_images_without_gt=False)

# Get the number of samples in the training and validations datasets.
train_dataset_size = train_dataset.get_dataset_size()
val_dataset_size = val_dataset.get_dataset_size()

print("Number of images in the training dataset:\t{:>6}".format(train_dataset_size))
print("Number of images in the validation dataset:\t{:>6}".format(val_dataset_size))

# visualize the input if needed
if train_dataset.show_images:
    for x, y in train_generator:
        pass

# 4. Set the remaining training parameters

# Define a learning rate schedule.

reduce_lr = ReduceLROnPlateau(patience=5, factor=0.666, verbose=1, epsilon=1e-1)

# Set the filepath under which you want to save the weights.
model_checkpoint = ModelCheckpoint(
    filepath='trained_models/drbox_epoch-{epoch:02d}_loss-{loss:.4f}_val_loss-{val_loss:.4f}.h5',
    monitor='val_loss',
    verbose=1,
    save_best_only=True,
    save_weights_only=False,
    mode='auto',
    period=1)

# model_checkpoint.best =

csv_logger = CSVLogger(filename='drbox_training_log.csv',
                       separator=',',
                       append=True)

terminate_on_nan = TerminateOnNaN()

callbacks = [model_checkpoint,
             csv_logger,
             reduce_lr,
             terminate_on_nan]

# 5. Train

# If you're resuming a previous training, set `initial_epoch` and `final_epoch` accordingly.
initial_epoch = 0
final_epoch = 100
steps_per_epoch = 1000

history = model.fit_generator(generator=train_generator,
                              steps_per_epoch=steps_per_epoch,
                              epochs=final_epoch,
                              callbacks=callbacks,
                              verbose=1,
                              validation_data=val_generator,
                              validation_steps=ceil(val_dataset_size / batch_size),
                              initial_epoch=initial_epoch)
