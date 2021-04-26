'''
A custom Keras layer to generate anchor boxes.

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
import keras.backend as K
from keras.engine.topology import InputSpec
from keras.engine.topology import Layer


class AnchorBoxes(Layer):
    '''
    A Keras layer to create an output tensor containing anchor box coordinates
    and variances based on the input tensor and the passed arguments.

    A set of 2D anchor boxes of different aspect ratios and angle of rotation is created 
    for each spatial unit of the input tensor. The number of anchor boxes created per 
    unit depends on the arguments `aspect_ratios` and angles. The boxes
    are parameterized by the coordinate tuple `(cx, cy, w, h)`.

    The purpose of having this layer in the network is to make the model self-sufficient
    at inference time. Since the model is predicting offsets to the anchor boxes
    (rather than predicting absolute box coordinates directly), one needs to know the anchor
    box coordinates in order to construct the final prediction boxes from the predicted offsets.
    If the model's output tensor did not contain the anchor box coordinates, the necessary
    information to convert the predicted offsets back to absolute coordinates would be missing
    in the model output.

    Input shape:
        4D tensor of shape or `(batch, height, width, channels)`

    Output shape:
        5D tensor of shape `(batch, height, width, n_boxes, 10)`. The last axis contains
        the five anchor box coordinates and the five variance values for each box.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 this_scale,
                 angles = [0, np.pi/6, np.pi/3, np.pi/2, 2*np.pi/3, 5*np.pi/6],
                 aspect_ratios=[2.0, 4.0, 6.0],
                 this_steps=None,
                 this_offsets=None,
                 variances=[0.1, 0.1, 0.2, 0.2],
                 normalize_coords=False,
                 **kwargs):
        '''
        All arguments need to be set to the same values as in the box encoding process, otherwise the behavior is undefined.
        Some of these arguments are explained in more detail in the documentation of the `DRBoxEncoder` class.

        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generated anchor boxes
                as a fraction of the shorter side of the input image.
            angles (list, optional): A list of floats >0 containing different angles for wich to create corresponding anchor boxes in radians.
            aspect_ratios (list, optional): The list of aspect ratios for which default boxes are to be
                generated for this layer.
            variances (list, optional): A list of 5 floats >0. The anchor box offset for each coordinate will be divided by
                its respective variance value.
            normalize_coords (bool, optional): Set to `True` if the model uses relative instead of absolute coordinates,
                i.e. if the model predicts box coordinates within [0,1] instead of absolute coordinates.
        '''
        if K.backend() != 'tensorflow':
            raise TypeError("This layer only supports TensorFlow at the moment, but you are using the {} backend.".format(K.backend()))

        if (this_scale < 0) or (this_scale > 1):
            raise ValueError("`this_scale` must be in [0, 1] but `this_scale` == {}".format(this_scale))

        if len(variances) != 5:
            raise ValueError("5 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        self.img_height = img_height
        self.img_width = img_width
        self.this_scale = this_scale
        self.aspect_ratios = aspect_ratios
        self.this_steps = this_steps
        self.this_offsets = this_offsets
        self.variances = variances
        self.normalize_coords = normalize_coords
        self.angles = angles
        # Compute the number of boxes per cell
        self.n_boxes = len(aspect_ratios)*len(angles)
        super(AnchorBoxes, self).__init__(**kwargs)

    def build(self, input_shape):
        self.input_spec = [InputSpec(shape=input_shape)]
        super(AnchorBoxes, self).build(input_shape)

    def call(self, x, mask=None):
        '''
        Return an anchor box tensor based on the shape of the input tensor.

        Note that this tensor does not participate in any graph computations at runtime. It is being created
        as a constant once during graph creation and is just being output along with the rest of the model output
        during runtime. Because of this, all logic is implemented as Numpy array operations and it is sufficient
        to convert the resulting Numpy array into a Keras tensor at the very end before outputting it.

        Arguments:
            x (tensor): 4D tensor of shape `(batch, height, width, channels)`. The input for this
                layer must be the output of the localization predictor layer.
        '''

        # Compute box width and height for each aspect ratio
        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios
        wha_list = []
        for ar in self.aspect_ratios:
            box_height = self.this_scale * size / np.sqrt(ar)
            box_width = self.this_scale * size * np.sqrt(ar)
            for angle in self.angles:
                wha_list.append((box_width, box_height, angle))
        wha_list = np.array(wha_list)

        # We need the shape of the input tensor
        batch_size, feature_map_height, feature_map_width, feature_map_channels = x.shape

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (self.this_steps is None):
            step_height = self.img_height / feature_map_height
            step_width = self.img_width / feature_map_width
        else:
            if isinstance(self.this_steps, (list, tuple)) and (len(self.this_steps) == 2):
                step_height = self.this_steps[0]
                step_width = self.this_steps[1]
            elif isinstance(self.this_steps, (int, float)):
                step_height = self.this_steps
                step_width = self.this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (self.this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(self.this_offsets, (list, tuple)) and (len(self.this_offsets) == 2):
                offset_height = self.this_offsets[0]
                offset_width = self.this_offsets[1]
            elif isinstance(self.this_offsets, (int, float)):
                offset_height = self.this_offsets
                offset_width = self.this_offsets

        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_height - 1) * step_height, feature_map_height)
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_width - 1) * step_width, feature_map_width)
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 5)`
        # where the last dimension will contain `(cx, cy, w, h, angle)`
        boxes_tensor = np.zeros((feature_map_height, feature_map_width, self.n_boxes, 5))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, self.n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, self.n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wha_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wha_list[:, 1] # Set h
        boxes_tensor[:, :, :, 4] = wha_list[:, 2] # Set angle

        # If `normalize_coords` is enabled, normalize the width and height to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, 0] /= self.img_width
            boxes_tensor[:, :, :, 1] /= self.img_height
            boxes_tensor[:, :, :, 2] /= self.img_width*(np.cos(boxes_tensor[:, :, :, 4])**2) + self.img_height*(np.sin(boxes_tensor[:, :, :, 4])**2)
            boxes_tensor[:, :, :, 3] /= self.img_height*(np.cos(boxes_tensor[:, :, :, 4])**2) + self.img_width*(np.sin(boxes_tensor[:, :, :, 4]) **2)

        # Create a tensor to contain the variances and append it to `boxes_tensor`. This tensor has the same shape 
        # as `boxes_tensor` and simply contains the same 5 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor) # Has shape `(feature_map_height, feature_map_width, n_boxes, 5)`
        variances_tensor += self.variances # Long live broadcasting
        # Now `boxes_tensor` becomes a tensor of shape `(feature_map_height, feature_map_width, n_boxes, 10)`
        boxes_tensor = np.concatenate((boxes_tensor, variances_tensor), axis=-1)

        # Now prepend one dimension to `boxes_tensor` to account for the batch size and tile it along
        # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 10)`
        boxes_tensor = np.expand_dims(boxes_tensor, axis=0)
        boxes_tensor = K.tile(K.constant(boxes_tensor, dtype='float32'), (K.shape(x)[0], 1, 1, 1, 1))

        return boxes_tensor

    def compute_output_shape(self, input_shape):
        batch_size, feature_map_height, feature_map_width, feature_map_channels = input_shape

        return (batch_size, feature_map_height, feature_map_width, self.n_boxes, 10)

    def get_config(self):
        config = {
            'img_height': self.img_height,
            'img_width': self.img_width,
            'this_scale': self.this_scale,
            'angles' : self.angles,
            'aspect_ratios': list(self.aspect_ratios),
            'variances': list(self.variances),
            'normalize_coords': self.normalize_coords
        }
        base_config = super(AnchorBoxes, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
