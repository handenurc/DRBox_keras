'''
An encoder that converts ground truth annotations to SSD-compatible training targets.

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

from bounding_box_utils.bounding_box_utils import ARiou180
from drbox_encoder_decoder.matching_utils import match_bipartite_greedy, match_multi

class DRBoxInputEncoder:
    '''
    Transforms ground truth labels for object detection in images
    (2D bounding box coordinates and class labels) to the format required for
    training an DRBox model.

    In the process of encoding the ground truth labels, a template of anchor boxes
    is being built, which are subsequently matched to the ground truth boxes
    via an intersection-over-union threshold criterion.
    '''

    def __init__(self,
                 img_height,
                 img_width,
                 n_classes,
                 predictor_sizes,
                 angles_global = [0,np.pi/6,np.pi/3,np.pi/2,2*np.pi/3,5*np.pi/6],
                 angles_per_layer = None,
                 min_scale=0.1,
                 max_scale=0.9,
                 scales=None,
                 aspect_ratios_global=[2.0, 4.0],
                 aspect_ratios_per_layer=None,
                 steps=None,
                 offsets=None,
                 variances=[0.1, 0.1, 0.2, 0.2, 0.2],
                 matching_type='multi',
                 pos_ariou_threshold=0.5,
                 neg_ariou_limit=0.3,
                 normalize_coords=True,
                 background_id=0,
                 batch_size=None):
        '''
        Arguments:
            img_height (int): The height of the input images.
            img_width (int): The width of the input images.
            n_classes (int): The number of positive classes, e.g. 20 for Pascal VOC, 80 for MS COCO.
            predictor_sizes (list): A list of int-tuples of the format `(height, width)`
                containing the output heights and widths of the convolutional predictor layers.
            angles_global (list, optional): A list of floats >0 containing different angles for wich to create
                corresponding anchor boxes in radians.
            angles_per_layer (list, optional): A list containing one angles list for each prediction layer in radians.
                If a list is passed, it overrides `angles_global`. Defaults to `None`. Note that you should
                set the angles such that the resulting anchor box shapes very roughly correspond to the shapes of the
                objects you are trying to detect.
            min_scale (float, optional): The smallest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect. Must be >0.
            max_scale (float, optional): The largest scaling factor for the size of the anchor boxes as a fraction
                of the shorter side of the input images. All scaling factors between the smallest and the
                largest will be linearly interpolated. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect. Must be greater than or equal to `min_scale`.
            scales (list, optional): A list of floats >0 containing scaling factors per convolutional predictor layer.
                The `k`th element is the scaling factors for the `k`th predictor layers. 
                Defaults to `None`. If a list is passed, this argument overrides `min_scale` and
                `max_scale`. All scaling factors must be greater than zero. Note that you should set the scaling factors
                such that the resulting anchor box sizes correspond to the sizes of the objects you are trying
                to detect.
            aspect_ratios_global (list, optional): The list of aspect ratios for which anchor boxes are to be
                generated. This list is valid for all prediction layers. Defaults to [0.5, 1.0, 2.0]. Note that you
                should set the aspect ratios such that the resulting anchor box shapes roughly correspond to the shapes
                of the objects you are trying to detect.
            aspect_ratios_per_layer (list, optional): A list containing one aspect ratio list for each prediction layer.
                If a list is passed, it overrides `aspect_ratios_global`. Defaults to `None`. Note that you should
                set the aspect ratios such that the resulting anchor box shapes very roughly correspond to the shapes of
                the objects you are trying to detect.
            steps (list, optional): `None` or a list with as many elements as there are predictor layers. The elements
                can be either ints/floats or tuples of two ints/floats. These numbers represent for each predictor layer
                how many pixels apart the anchor box center points should be vertically and horizontally along the
                spatial grid over the image. If the list contains ints/floats, then that value will be used for both
                spatial dimensions. If the list contains tuples of two ints/floats, then they represent
                `(step_height, step_width)`. If no steps are provided, then they will be computed such that the anchor
                box center points will form an equidistant grid within the image dimensions.
            offsets (list, optional): `None` or a list with as many elements as there are predictor layers. The elements
                can be either floats or tuples of two floats. These numbers represent for each predictor layer how many
                pixels from the top and left boarders of the image the top-most and left-most anchor box center points
                should be as a fraction of `steps`. The last bit is important: The offsets are not absolute pixel
                values, but fractions of the step size specified in the `steps` argument. If the list contains floats,
                then that value will be used for both spatial dimensions. If the list contains tuples of two floats,
                then they represent `(vertical_offset, horizontal_offset)`. If no offsets are provided, then they will
                default to 0.5 of the step size.
            variances (list, optional): A list of 5 floats >0. The anchor box offset for each coordinate will be divided
                by its respective variance value.
            matching_type (str, optional): Can be either 'multi' or 'bipartite'. In 'bipartite' mode, each ground truth
                box will be matched only to the one anchor box with the highest IoU overlap. In 'multi' mode, in
                addition to the aforementioned bipartite matching, all anchor boxes with an IoU overlap greater than or
                equal to the `pos_iou_threshold` will be matched to a given ground truth box.
            pos_iou_threshold (float, optional): The intersection-over-union similarity threshold that must be
                met in order to match a given ground truth box to a given anchor box.
            neg_ariou_limit (float, optional): The maximum allowed intersection-over-union similarity of an
                anchor box with any ground truth box to be labeled a negative (i.e. background) box. If an
                anchor box is neither a positive, nor a negative box, it will be ignored during training.
            normalize_coords (bool, optional): If `True`, the encoder uses relative instead of absolute coordinates.
                This means instead of using absolute tartget coordinates, the encoder will scale all coordinates to be
                within [0,1]. This way learning becomes independent of the input image size.
            background_id (int, optional): Determines which class ID is for the background class.
            batch_size (int, optional): the batch size of the generator, if None, the batch_size will be infered from
                the ground_truth_labels
        '''
        predictor_sizes = np.array(predictor_sizes)
        if predictor_sizes.ndim == 1:
            predictor_sizes = np.expand_dims(predictor_sizes, axis=0)

        ##################################################################################
        # Handle exceptions.
        ##################################################################################

        if (min_scale is None or max_scale is None) and scales is None:
            raise ValueError("Either `min_scale` and `max_scale` or `scales` need to be specified.")

        if scales:
            if (len(scales) != predictor_sizes.shape[0] ): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either scales is None or len(scales) == len(predictor_sizes), but len(scales) == {} and len(predictor_sizes) == {}".format(len(scales), len(predictor_sizes)))
            scales = np.array(scales)
            if np.any(scales <= 0):
                raise ValueError("All values in `scales` must be greater than 0, but the passed list of scales is {}".format(scales))
        else: # If no list of scales was passed, we need to make sure that `min_scale` and `max_scale` are valid values.
            if not 0 < min_scale <= max_scale:
                raise ValueError("It must be 0 < min_scale <= max_scale, but it is min_scale = {} and max_scale = {}".format(min_scale, max_scale))

        if not (aspect_ratios_per_layer is None):
            if (len(aspect_ratios_per_layer) != predictor_sizes.shape[0]): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either aspect_ratios_per_layer is None or len(aspect_ratios_per_layer) == len(predictor_sizes), but len(aspect_ratios_per_layer) == {} and len(predictor_sizes) == {}".format(len(aspect_ratios_per_layer), len(predictor_sizes)))
            for aspect_ratios in aspect_ratios_per_layer:
                if np.any(np.array(aspect_ratios) <= 0):
                    raise ValueError("All aspect ratios must be greater than zero.")
        else:
            if (aspect_ratios_global is None):
                raise ValueError("At least one of `aspect_ratios_global` and `aspect_ratios_per_layer` must not be `None`.")
            if np.any(np.array(aspect_ratios_global) <= 0):
                raise ValueError("All aspect ratios must be greater than zero.")

        if not (angles_per_layer is None):
            if (len(angles_per_layer) != predictor_sizes.shape[0]): # Must be two nested `if` statements since `list` and `bool` cannot be combined by `&`
                raise ValueError("It must be either angles_per_layer is None or len(angles_per_layer) == len(predictor_sizes), but len(angles_per_layer) == {} and len(predictor_sizes) == {}".format(len(angles_per_layer), len(predictor_sizes)))

        else:
            if (angles_global is None):
                raise ValueError("At least one of `angles_global` and `angles_per_layer` must not be `None`.")

        if len(variances) != 5:
            raise ValueError("5 variance values must be pased, but {} values were received.".format(len(variances)))
        variances = np.array(variances)
        if np.any(variances <= 0):
            raise ValueError("All variances must be >0, but the variances given are {}".format(variances))

        if (not (steps is None)) and (len(steps) != predictor_sizes.shape[0]):
            raise ValueError("You must provide at least one step value per predictor layer.")

        if (not (offsets is None)) and (len(offsets) != predictor_sizes.shape[0]):
            raise ValueError("You must provide at least one offset value per predictor layer.")

        ##################################################################################
        # Set or compute members.
        ##################################################################################

        self.img_height = img_height
        self.img_width = img_width
        self.n_classes = n_classes + 1 # + 1 for the background class
        self.predictor_sizes = predictor_sizes
        self.min_scale = min_scale
        self.max_scale = max_scale

        # If `scales` is None, compute the scaling factors by linearly interpolating between
        # `min_scale` and `max_scale`. If an explicit list of `scales` is given, however,
        # then it takes precedent over `min_scale` and `max_scale`.
        if (scales is None):
            self.scales = np.linspace(self.min_scale, self.max_scale, len(self.predictor_sizes))
        else:
            # If a list of scales is given explicitly, we'll use that instead of computing it from `min_scale` and `max_scale`.
            self.scales = scales

        # If `aspect_ratios_per_layer` is None, then we use the same list of aspect ratios
        # `aspect_ratios_global` for all predictor layers. If `aspect_ratios_per_layer` is given,
        # however, then it takes precedent over `aspect_ratios_global`.
        if (aspect_ratios_per_layer is None):
            self.aspect_ratios = [aspect_ratios_global] * predictor_sizes.shape[0]
        else:
            # If aspect ratios are given per layer, we'll use those.
            self.aspect_ratios = aspect_ratios_per_layer

        # If `angles_per_layer` is None, then we use the same list of angles
        # `angles_global` for all predictor layers. If `angles_per_layer` is given,
        # however, then it takes precedent over `angles_global`.
        if (angles_per_layer is None):
            self.angles = [angles_global] * predictor_sizes.shape[0]
        else:
            # If angles are given per layer, we'll use those.
            self.angles = angles_per_layer

        if not (steps is None):
            self.steps = steps
        else:
            self.steps = [None] * predictor_sizes.shape[0]
        if not (offsets is None):
            self.offsets = offsets
        else:
            self.offsets = [None] * predictor_sizes.shape[0]

        self.variances = variances
        self.matching_type = matching_type
        self.pos_ariou_threshold = pos_ariou_threshold
        self.neg_ariou_limit = neg_ariou_limit
        self.normalize_coords = normalize_coords
        self.background_id = background_id
        self.batch_size = batch_size

        ##################################################################################
        # Compute the anchor boxes for each predictor layer.
        ##################################################################################

        # Compute the anchor boxes for each predictor layer. We only have to do this once
        # since the anchor boxes depend only on the model configuration, not on the input data.
        # For each predictor layer (i.e. for each scaling factor) the tensors for that layer's
        # anchor boxes will have the shape `(feature_map_height, feature_map_width, n_boxes, 5)`.

        self.boxes_list = [] # This will store the anchor boxes for each predicotr layer.

        # The following lists just store diagnostic information. Sometimes it's handy to have the
        # boxes' center points, heights, widths, etc. in a list.
        self.wha_list_diag = [] # Box widths, heights and angles for each predictor layer
        self.steps_diag = [] # Horizontal and vertical distances between any two boxes for each predictor layer
        self.offsets_diag = [] # Offsets for each predictor layer
        self.centers_diag = [] # Anchor box center points as `(cy, cx)` for each predictor layer

        # Iterate over all predictor layers and compute the anchor boxes for each one.
        for i in range(len(self.predictor_sizes)):
            boxes, center, wha, step, offset = self.generate_anchor_boxes_for_layer(feature_map_size=self.predictor_sizes[i],
                                                                                   aspect_ratios=self.aspect_ratios[i],
                                                                                   this_scale=self.scales[i],
                                                                                   angles=self.angles[i],
                                                                                   this_steps=self.steps[i],
                                                                                   this_offsets=self.offsets[i],
                                                                                   diagnostics=True)

            self.boxes_list.append(boxes)
            self.wha_list_diag.append(wha)
            self.steps_diag.append(step)
            self.offsets_diag.append(offset)
            self.centers_diag.append(center)

        if self.batch_size:
            # initializes the encoding template
            self.encoding_template = self.generate_encoding_template(batch_size=self.batch_size, diagnostics=False)
        else:
            # we initialize encoding template to an empty array
            self.encoding_template = []


    def __call__(self, ground_truth_labels, diagnostics=False):
        '''
        Converts ground truth bounding box data into a suitable format to train a DRBox model.

        Arguments:
            ground_truth_labels (list): A python list of length `batch_size` that contains one 2D Numpy array
                for each batch image. Each such array has `k` rows for the `k` ground truth bounding boxes belonging
                to the respective image, and the data for each ground truth bounding box has the format
                `(class_id, cx, cy, w, h, angle), and `class_id` must be
                an integer greater than 0 for all boxes as class ID 0 is reserved for the background class.
            diagnostics (bool, optional): If `True`, not only the encoded ground truth tensor will be returned,
                but also a copy of it with anchor box coordinates in place of the ground truth coordinates.
                This can be very useful if you want to visualize which anchor boxes got matched to which ground truth
                boxes.

        Returns:
            `y_encoded`, a 3D numpy array of shape `(batch_size, #boxes, #classes + 5 + 5 + 5)` that serves as the
            ground truth label tensor for training, where `#boxes` is the total number of boxes predicted by the
            model per image, and the classes are one-hot-encoded. The five elements after the class vectors in
            the last axis are the box coordinates, the next five elements after that are just dummy elements, and
            the last five elements are the variances.
        '''

        # Mapping to define which indices represent which coordinates in the ground truth.
        class_id = 0
        cx = 1
        cy = 2
        w = 3
        h = 4
        angle = 5

        # if the batch size has not been defined before or if the actual batch size of the labels is not the previously
        # defined batch_size, we change the value of the batch size and the encoding template
        if (self.batch_size is None) or (self.batch_size != len(ground_truth_labels)) or \
                (len(self.encoding_template) != self.batch_size):

            self.batch_size = len(ground_truth_labels)

            ##################################################################################
            # Generate the template for y_encoded.
            ##################################################################################
            self.encoding_template = self.generate_encoding_template(batch_size=self.batch_size, diagnostics=False)

        y_encoded = np.copy(self.encoding_template)

        ##################################################################################
        # Match ground truth boxes to anchor boxes.
        ##################################################################################

        # Match the ground truth boxes to the anchor boxes. Every anchor box that does not have
        # a ground truth match and for which the maximal IoU overlap with any ground truth box is less
        # than or equal to `neg_ariou_limit` will be a negative (background) box.

        y_encoded[:, :, self.background_id] = 1 # All boxes are background boxes by default.
        class_vectors = np.eye(self.n_classes) # An identity matrix that we'll use as one-hot class vectors

        for i in range(len(ground_truth_labels)): # For each batch item...

            if ground_truth_labels[i].size == 0: continue # If there is no ground truth for this batch item, there is nothing to match.
            labels = ground_truth_labels[i].astype(np.float) # The labels for this batch item

            # Check for degenerate ground truth bounding boxes before attempting any computations.
            if np.any(labels[:,[w]] <= 0) or np.any(labels[:,[h]] <= 0):
                raise DegenerateBoxError("SSDInputEncoder detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, labels) +
                                         "i.e. bounding boxes where h <=0 and/or w <= 0. Degenerate ground truth " +
                                         "bounding boxes will lead to NaN errors during the training.")

            if self.normalize_coords:
                labels[:,cx] /= self.img_width
                labels[:,cy] /= self.img_height
                labels[:,w] /= self.img_width*(np.cos(labels[:,angle])**2) + self.img_height*(np.sin(labels[:,angle])**2)
                labels[:,h] /= self.img_height*(np.cos(labels[:,angle])**2) + self.img_width*(np.sin(labels[:,angle]) **2)


            classes_one_hot = class_vectors[labels[:, class_id].astype(np.int)] # The one-hot class IDs for the ground truth boxes of this batch item
            labels_one_hot = np.concatenate([classes_one_hot, labels[:, [cx,cy,w,h,angle]]], axis=-1) # The one-hot version of the labels for this batch item

            # Compute the ARiou180 similarities between all anchor boxes and all ground truth boxes for this batch item.
            # This is a matrix of shape `(num_ground_truth_boxes, num_anchor_boxes)`.
            similarities = ARiou180(labels[:, [cx, cy, w, h, angle]], y_encoded[i, :, -15:-10])

            # First: Do bipartite matching, i.e. match each ground truth box to the one anchor box with the highest ARiou180.
            #        This ensures that each ground truth box will have at least one good match.

            # For each ground truth box, get the anchor box to match with it.

            bipartite_matches = match_bipartite_greedy(weight_matrix=similarities)

            # Write the ground truth data to the matched anchor boxes.
            y_encoded[i, bipartite_matches, :-10] = labels_one_hot

            # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
            similarities[:, bipartite_matches] = 0

            # Second: Maybe do 'multi' matching, where each remaining anchor box will be matched to its most similar
            #         ground truth box with an ARIou180 of at least `pos_ariou_threshold`, or not matched if there is no
            #         such ground truth box.

            if self.matching_type == 'multi':

                # Get all matches that satisfy the IoU threshold.
                matches = match_multi(weight_matrix=similarities, threshold=self.pos_ariou_threshold)
                # Write the ground truth data to the matched anchor boxes.
                y_encoded[i, matches[1], :-10] = labels_one_hot[matches[0]]

                # Set the columns of the matched anchor boxes to zero to indicate that they were matched.
                similarities[:, matches[1]] = 0

            # Third: Now after the matching is done, all negative (background) anchor boxes that have
            #        an IoU of `neg_ariou_limit` or more with any ground truth box will be set to neutral,
            #        i.e. they will no longer be background boxes. These anchors are "too close" to a
            #        ground truth box to be valid background boxes.
            
            max_background_similarities = np.amax(similarities, axis=0)
            neutral_boxes = np.nonzero(max_background_similarities >= self.neg_ariou_limit)[0]
            y_encoded[i, neutral_boxes, self.background_id] = 0

        ##################################################################################
        # Convert box coordinates to anchor box offsets.
        ##################################################################################

        y_encoded = np.swapaxes(y_encoded, 0, 2)

        cx_cy_gt = y_encoded[[-15, -14]]
        cx_cy_anchor = y_encoded[[-10, -9]]
        w_h_anchor = y_encoded[[-8, -7]]
        cx_cy_variance = y_encoded[[-5, -4]]

        cx_cy_gt -= cx_cy_anchor # cx(gt) - cx(anchor), cy(gt) - cy(anchor)
        cx_cy_gt /= w_h_anchor * cx_cy_variance  # (cx(gt) - cx(anchor)) / w(anchor) / cx_variance, (cy(gt) - cy(anchor)) / h(anchor) / cy_variance
        y_encoded[[-15, -14]] = cx_cy_gt

        w_h_encoded = y_encoded[[-13, -12]]
        w_h_variance = y_encoded[[-3, -2]]

        w_h_encoded /= w_h_anchor  # w(gt) / w(anchor), h(gt) / h(anchor)
        w_h_encoded = np.log(
            w_h_encoded) / w_h_variance  # ln(w(gt) / w(anchor)) / w_variance, ln(h(gt) / h(anchor)) / h_variance (ln == natural logarithm)
        y_encoded[[-13, -12]] = w_h_encoded

        angle_encoded = y_encoded[-11]
        angle_anchor = y_encoded[-6]

        angle_encoded -= angle_anchor  # angle (gt) - angle(anchor)
        angle_encoded = np.tan(angle_encoded)  # tan(angle (gt) - angle(anchor))
        y_encoded[-11] = angle_encoded

        y_encoded = np.swapaxes(y_encoded, 2, 0)

        if diagnostics:
            # Here we'll save the matched anchor boxes (i.e. anchor boxes that were matched to a ground truth box, but keeping the anchor box coordinates).
            y_matched_anchors = np.copy(y_encoded)
            y_matched_anchors[:,:,-15:-10] = 0 # Keeping the anchor box coordinates means setting the offsets to zero.
            return y_encoded, y_matched_anchors
        else:
            return y_encoded

    def generate_anchor_boxes_for_layer(self,
                                        feature_map_size,
                                        aspect_ratios,
                                        this_scale,
                                        angles,
                                        this_steps=None,
                                        this_offsets=None,
                                        diagnostics=False):
        '''
        Computes an array of the spatial positions and sizes of the anchor boxes for one predictor layer
        of size `feature_map_size == [feature_map_height, feature_map_width]`.

        Arguments:
            feature_map_size (tuple): A list or tuple `[feature_map_height, feature_map_width]` with the spatial
                dimensions of the feature map for which to generate the anchor boxes.
            aspect_ratios (list): A list of floats, the aspect ratios for which anchor boxes are to be generated.
                All list elements must be unique.
            this_scale (float): A float in [0, 1], the scaling factor for the size of the generate anchor boxes
                as a fraction of the shorter side of the input image.
            angles (list): a list of floats , the angles for wich anchor boxes are to be generated.
                All lists elements must be unique.
            diagnostics (bool, optional): If true, the following additional outputs will be returned:
                1) A list of the center point `x` and `y` coordinates for each spatial location.
                2) A list containing `(width, height, angle)` for each box aspect ratio.
                3) A tuple containing `(step_height, step_width)`
                4) A tuple containing `(offset_height, offset_width)`
                This information can be useful to understand in just a few numbers what the generated grid of
                anchor boxes actually looks like, i.e. how large the different boxes are and how dense
                their spatial distribution is, in order to determine whether the box grid covers the input images
                appropriately and whether the box sizes are appropriate to fit the sizes of the objects
                to be detected.

        Returns:
            A 4D Numpy tensor of shape `(feature_map_height, feature_map_width, n_boxes_per_cell, 5)` where the
            last dimension contains `(cx, cy, w, h, angle)` for each anchor box in each cell of the feature map.
        '''
        # Compute box width and height for each aspect ratio.

        # The shorter side of the image will be used to compute `w` and `h` using `scale` and `aspect_ratios`.
        size = min(self.img_height, self.img_width)
        # Compute the box widths and and heights for all aspect ratios

        wha_list = []
        for ar in aspect_ratios:
            box_width = this_scale * size * np.sqrt(ar)
            box_height = this_scale * size / np.sqrt(ar)
            for angle in angles:
                wha_list.append((box_width, box_height, angle))
        wha_list = np.array(wha_list)
        n_boxes = len(wha_list)

        # Compute the grid of box center points. They are identical for all aspect ratios.

        # Compute the step sizes, i.e. how far apart the anchor box center points will be vertically and horizontally.
        if (this_steps is None):
            step_height = self.img_height / feature_map_size[0]
            step_width = self.img_width / feature_map_size[1]
        else:
            if isinstance(this_steps, (list, tuple)) and (len(this_steps) == 2):
                step_height = this_steps[0]
                step_width = this_steps[1]
            elif isinstance(this_steps, (int, float)):
                step_height = this_steps
                step_width = this_steps
        # Compute the offsets, i.e. at what pixel values the first anchor box center point will be from the top and from the left of the image.
        if (this_offsets is None):
            offset_height = 0.5
            offset_width = 0.5
        else:
            if isinstance(this_offsets, (list, tuple)) and (len(this_offsets) == 2):
                offset_height = this_offsets[0]
                offset_width = this_offsets[1]
            elif isinstance(this_offsets, (int, float)):
                offset_height = this_offsets
                offset_width = this_offsets

        # Now that we have the offsets and step sizes, compute the grid of anchor box center points.
        cy = np.linspace(offset_height * step_height, (offset_height + feature_map_size[0] - 1) * step_height, feature_map_size[0])
        cx = np.linspace(offset_width * step_width, (offset_width + feature_map_size[1] - 1) * step_width, feature_map_size[1])
        cx_grid, cy_grid = np.meshgrid(cx, cy)
        cx_grid = np.expand_dims(cx_grid, -1) # This is necessary for np.tile() to do what we want further down
        cy_grid = np.expand_dims(cy_grid, -1) # This is necessary for np.tile() to do what we want further down

        # Create a 4D tensor template of shape `(feature_map_height, feature_map_width, n_boxes, 5)`
        # where the last dimension will contain `(cx, cy, w, h, angle)`
        boxes_tensor = np.zeros((feature_map_size[0], feature_map_size[1], n_boxes, 5))

        boxes_tensor[:, :, :, 0] = np.tile(cx_grid, (1, 1, n_boxes)) # Set cx
        boxes_tensor[:, :, :, 1] = np.tile(cy_grid, (1, 1, n_boxes)) # Set cy
        boxes_tensor[:, :, :, 2] = wha_list[:, 0] # Set w
        boxes_tensor[:, :, :, 3] = wha_list[:, 1] # Set h
        boxes_tensor[:, :, :, 4] = wha_list[:, 2] # Set angle

        # `normalize_coords` is enabled, normalize the coordinates to be within [0,1]
        if self.normalize_coords:
            boxes_tensor[:, :, :, 0] /= self.img_width
            boxes_tensor[:, :, :, 1] /= self.img_height
            boxes_tensor[:, :, :, 2] /= self.img_width*(np.cos(boxes_tensor[:, :, :, 4])**2) + self.img_height*(np.sin(boxes_tensor[:, :, :, 4])**2)
            boxes_tensor[:, :, :, 3] /= self.img_height*(np.cos(boxes_tensor[:, :, :, 4])**2) + self.img_width*(np.sin(boxes_tensor[:, :, :, 4]) **2)

        if diagnostics:
            return boxes_tensor, (cy, cx), wha_list, (step_height, step_width), (offset_height, offset_width)
        else:
            return boxes_tensor

    def generate_encoding_template(self, batch_size, diagnostics=False):
        '''
        Produces an encoding template for the ground truth label tensor for a given batch.

        Note that all tensor creation, reshaping and concatenation operations performed in this function
        and the sub-functions it calls are identical to those performed inside the DRBox model. This, of course,
        must be the case in order to preserve the spatial meaning of each box prediction, but it's useful to make
        yourself aware of this fact and why it is necessary.

        In other words, the boxes in `y_encoded` must have a specific order in order correspond to the right spatial
        positions and scales of the boxes predicted by the model. The sequence of operations here ensures that
        `y_encoded` has this specific form.

        Arguments:
            batch_size (int): The batch size.
            diagnostics (bool, optional): See the documnentation for `generate_anchor_boxes()`. The diagnostic output
                here is similar, just for all predictor conv layers.

        Returns:
            A Numpy array of shape `(batch_size, #boxes, #classes + 15)`, the template into which to encode
            the ground truth labels for training. The last axis has length `#classes + 15` because the model
            output contains not only the 5 predicted box coordinate offsets, but also the 5 coordinates for
            the anchor boxes and the 5 variance values.
        '''
        
        # Tile the anchor boxes for each predictor layer across all batch items.
        boxes_batch = []
        for boxes in self.boxes_list:
            # Prepend one dimension to `self.boxes_list` to account for the batch size and tile it along.
            # The result will be a 5D tensor of shape `(batch_size, feature_map_height, feature_map_width, n_boxes, 5)`
            boxes = np.expand_dims(boxes, axis=0)
            boxes = np.tile(boxes, (batch_size, 1, 1, 1, 1))

            # Now reshape the 5D tensor above into a 3D tensor of shape
            # `(batch, feature_map_height * feature_map_width * n_boxes, 5)`. The resulting
            # order of the tensor content will be identical to the order obtained from the reshaping operation
            # in our Keras model (we're using the Tensorflow backend, and tf.reshape() and np.reshape()
            # use the same default index order, which is C-like index ordering)
            boxes = np.reshape(boxes, (batch_size, -1, 5))
            boxes_batch.append(boxes)

        # Concatenate the anchor tensors from the individual layers to one.
        boxes_tensor = np.concatenate(boxes_batch, axis=1)

        # 3: Create a template tensor to hold the one-hot class encodings of shape `(batch, #boxes, #classes)`
        #    It will contain all zeros for now, the classes will be set in the matching process that follows
        classes_tensor = np.zeros((batch_size, boxes_tensor.shape[1], self.n_classes))

        # 4: Create a tensor to contain the variances. This tensor has the same shape as `boxes_tensor` and simply
        #    contains the same 5 variance values for every position in the last axis.
        variances_tensor = np.zeros_like(boxes_tensor)
        variances_tensor += self.variances # Long live broadcasting

        # 4: Concatenate the classes, boxes and variances tensors to get our final template for y_encoded. We also need
        #    another tensor of the shape of `boxes_tensor` as a space filler so that `y_encoding_template` has the same
        #    shape as the SSD model output tensor. The content of this tensor is irrelevant, we'll just use
        #    `boxes_tensor` a second time.
        y_encoding_template = np.concatenate((classes_tensor, boxes_tensor, boxes_tensor, variances_tensor), axis=2)

        if diagnostics:
            return y_encoding_template, self.centers_diag, self.wh_list_diag, self.steps_diag, self.offsets_diag
        else:
            return y_encoding_template

class DegenerateBoxError(Exception):
    '''
    An exception class to be raised if degenerate boxes are being detected.
    '''
    pass
