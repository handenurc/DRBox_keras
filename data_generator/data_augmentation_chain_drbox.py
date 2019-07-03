'''
A data augmentation pipeline for datasets in bird's eye view, i.e. where there is
no "up" or "down" in the images.

Copyright (C) 2018 Pierluigi Ferrari

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Modifications author : Paul Pontisso
'''

from __future__ import division
import numpy as np

from data_generator.object_detection_2d_photometric_ops import ConvertColor, ConvertDataType, ConvertTo3Channels, \
    RandomBrightness, RandomContrast, RandomHue, RandomSaturation
from data_generator.object_detection_2d_geometric_ops import Resize, RandomFlip, RandomRotate, RandomTranslate, \
    RandomScale
from data_generator.object_detection_2d_patch_sampling_ops import PatchCoordinateGenerator, RandomPatch
from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter, ImageValidator


# added random_rotate_small, RandomScale and RandomRotate functions

class DataAugmentation:
    '''
    A data augmentation pipeline for datasets in bird's eye view, i.e. where there is
    no "up" or "down" in the images.

    Applies a chain of photometric and geometric image transformations. For documentation, please refer
    to the documentation of the individual transformations involved.
    '''

    def __init__(self,
                 resize_height,
                 resize_width,
                 random_brightness=(-20, 20, 0.5),
                 random_contrast=(0.8, 1.0, 0.5),
                 random_saturation=(0.8, 1.8, 0.5),
                 random_hue=(10, 0.5),
                 random_flip=0.5,
                 random_rotate_small=([np.pi / 40, np.pi / 30], 0.5),
                 random_rotate_big=([np.pi / 2, np.pi, 3 * np.pi / 2], 0.5),
                 min_scale=0.8,
                 max_scale=1.05,
                 min_aspect_ratio=0.8,
                 max_aspect_ratio=1.2,
                 n_trials_max=3,
                 overlap_criterion='center_point',
                 bounds_box_filter=(0.3, 1.0),
                 bounds_validator=(0.5, 1.0),
                 n_boxes_min=1,
                 random_translate=((0.03, 0.05), (0.03, 0.05), 0.5),
                 random_scale=(0.9, 1.1, 0.5),
                 proba_no_aug=1/3):

        self.n_trials_max = n_trials_max
        self.overlap_criterion = overlap_criterion
        self.bounds_box_filter = bounds_box_filter
        self.bounds_validator = bounds_validator
        self.n_boxes_min = n_boxes_min

        self.proba_no_aug = proba_no_aug # the probability of not performing any transformations

        # Determines which boxes are kept in an image after the transformations have been applied.
        self.box_filter = BoxFilter(check_overlap=True,
                                    check_min_area=False,
                                    check_degenerate=False,
                                    overlap_criterion=self.overlap_criterion,
                                    overlap_bounds=self.bounds_box_filter)

        self.box_filter_resize = BoxFilter(check_overlap=False,
                                           check_min_area=True,
                                           check_degenerate=True,
                                           min_area=16)

        # Determines whether the result of the transformations is a valid training image.
        self.image_validator = ImageValidator(overlap_criterion=self.overlap_criterion,
                                              bounds=self.bounds_validator,
                                              n_boxes_min=self.n_boxes_min)

        # Utility transformations
        self.convert_to_3_channels = ConvertTo3Channels()  # Make sure all images end up having 3 channels.
        self.convert_RGB_to_HSV = ConvertColor(current='RGB', to='HSV')
        self.convert_HSV_to_RGB = ConvertColor(current='HSV', to='RGB')
        self.convert_to_float32 = ConvertDataType(to='float32')
        self.convert_to_uint8 = ConvertDataType(to='uint8')
        self.resize = Resize(height=resize_height,
                             width=resize_width,
                             box_filter=self.box_filter_resize)

        # Photometric transformations
        self.random_brightness = RandomBrightness(lower=random_brightness[0], upper=random_brightness[1],
                                                  prob=random_brightness[2])
        self.random_contrast = RandomContrast(lower=random_contrast[0], upper=random_contrast[1],
                                              prob=random_contrast[2])
        self.random_saturation = RandomSaturation(lower=random_saturation[0], upper=random_saturation[1],
                                                  prob=random_saturation[2])
        self.random_hue = RandomHue(max_delta=random_hue[0], prob=random_hue[1])

        # Geometric transformations
        self.random_horizontal_flip = RandomFlip(dim='horizontal', prob=random_flip)
        self.random_vertical_flip = RandomFlip(dim='vertical', prob=random_flip)
        self.random_translate = RandomTranslate(dy_minmax=random_translate[0],
                                                dx_minmax=random_translate[1],
                                                prob=random_translate[2],
                                                box_filter=self.box_filter,
                                                image_validator=self.image_validator,
                                                n_trials_max=self.n_trials_max)

        self.random_rotate_small = RandomRotate(angles=random_rotate_small[0],
                                                prob=random_rotate_small[1],
                                                box_filter=self.box_filter,
                                                image_validator=self.image_validator,
                                                n_trials_max=self.n_trials_max)

        self.random_rotate_big = RandomRotate(angles=random_rotate_big[0],
                                              prob=random_rotate_big[1],
                                              box_filter=self.box_filter,
                                              image_validator=self.image_validator,
                                              n_trials_max=self.n_trials_max)

        self.random_zoom_in = RandomScale(min_factor=1.0,
                                          max_factor=random_scale[1],
                                          prob=random_scale[2],
                                          box_filter=self.box_filter,
                                          image_validator=self.image_validator,
                                          n_trials_max=self.n_trials_max)

        self.random_zoom_out = RandomScale(min_factor=random_scale[0],
                                           max_factor=random_scale[0],
                                           prob=random_scale[2],
                                           box_filter=self.box_filter,
                                           image_validator=self.image_validator,
                                           n_trials_max=self.n_trials_max)

        # random patch generator is not used for the moment but it could be useful in your project
        self.patch_coord_generator = PatchCoordinateGenerator(must_match='h_w',
                                                              min_scale=min_scale,
                                                              max_scale=max_scale,
                                                              scale_uniformly=False,
                                                              min_aspect_ratio=min_aspect_ratio,
                                                              max_aspect_ratio=max_aspect_ratio)

        self.random_patch = RandomPatch(patch_coord_generator=self.patch_coord_generator,
                                        box_filter=self.box_filter,
                                        image_validator=self.image_validator,
                                        n_trials_max=self.n_trials_max,
                                        prob=0.5,
                                        can_fail=False)

        # If we zoom in, do translation before scaling.
        self.sequence1 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.random_horizontal_flip,
                          self.random_vertical_flip,
                          self.random_translate,
                          self.random_rotate_big,
                          self.random_rotate_small,
                          self.random_zoom_in,
                          self.random_patch,
                          self.resize]

        # If we zoom out, do translation after scaling.
        self.sequence2 = [self.convert_to_3_channels,
                          self.convert_to_float32,
                          self.random_brightness,
                          self.random_contrast,
                          self.convert_to_uint8,
                          self.convert_RGB_to_HSV,
                          self.convert_to_float32,
                          self.random_saturation,
                          self.random_hue,
                          self.convert_to_uint8,
                          self.convert_HSV_to_RGB,
                          self.random_horizontal_flip,
                          self.random_vertical_flip,
                          self.random_zoom_out,
                          self.random_translate,
                          self.random_rotate_big,
                          self.random_rotate_small,
                          self.random_patch,
                          self.resize]

    def __call__(self, image, labels=None):

        # Choose to perform data augmentation or not
        rand = np.random.choice(int(1/self.proba_no_aug))

        if rand == 0:

            rand = np.random.choice(2)

            # Choose sequence 2 with probability 0.5
            if rand == 0:
                if not (labels is None):
                    for transform in self.sequence1:
                        image, labels = transform(image, labels)
                    return image, labels
                else:
                    for transform in self.sequence1:
                        image = transform(image)
                    return image

            # Choose sequence 2 with probability 0.5
            elif rand == 1:
                if not (labels is None):
                    for transform in self.sequence2:
                        image, labels = transform(image, labels)
                    return image, labels
                else:
                    for transform in self.sequence2:
                        image = transform(image)
                    return image

        # Do not perform any transformations
        else:
            if not (labels is None):
                return image, labels
            else:
                return image
