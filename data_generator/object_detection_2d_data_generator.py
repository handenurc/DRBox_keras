'''
A data generator for 2D object detection.

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
import warnings
import sklearn.utils
from copy import deepcopy
from PIL import Image
import csv
import os
import sys
from tqdm import tqdm
from bounding_box_utils.visualization import visualize

from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter

try:
    import pickle
except ImportError:
    warnings.warn("'pickle' module is missing. You won't be able to save parsed file lists and annotations as pickled files.")


from data_generator.object_detection_2d_image_boxes_validation_utils import BoxFilter

class DegenerateBatchError(Exception):
    '''
    An exception class to be raised if a generated batch ends up being degenerate,
    e.g. if a generated batch is empty.
    '''
    pass

class DatasetError(Exception):
    '''
    An exception class to be raised if a anything is wrong with the dataset,
    in particular if you try to generate batches when no dataset was loaded.
    '''
    pass

# just one way to read data : from csv files
class DataGenerator:
    '''
    A generator to generate batches of samples and corresponding labels indefinitely.

    Can shuffle the dataset consistently after each complete pass.

    A general-purpose CSV parser is provided to parse annotation data,

    Can perform image transformations for data conversion and data augmentation,
    for details please refer to the documentation of the `generate()` method.
    '''

    # only parser for csv files
    def __init__(self,
                 load_images_into_memory=False,
                 filenames=None,
                 images_dir=None,
                 labels=None,
                 image_ids=None,
                 verbose=True,
                 show_images=False):
        '''
        This class provides a parser method that you call separately after calling the constructor to assemble
        the list of image filenames and the list of labels for the dataset from CSV. If you already
        have the image filenames and labels in a suitable format (see argument descriptions below), you can pass
        them right here in the constructor, in which case you do not need to call any of the parser methods afterwards.

        In case you would like not to load any labels at all, simply pass a list of image filenames here.

        Arguments:
            load_images_into_memory (bool, optional): If `True`, the entire dataset will be loaded into memory.
                This enables noticeably faster data generation than loading batches of images into memory ad hoc.
                Be sure that you have enough memory before you activate this option.
            filenames (string or list, optional): `None` or either a Python list/tuple or a string representing
                a filepath. If a list/tuple is passed, it must contain the file names (full paths) of the
                images to be used. Note that the list/tuple must contain the paths to the images,
                not the images themselves. If a filepath string is passed, it must point to
                a text file. Each line of the text file contains the file name (basename of the file only,
                not the full directory path) to one image and nothing else.
            images_dir (string, optional): In case a text file is passed for `filenames`, the full paths to
                the images will be composed from `images_dir` and the names in the text file, i.e. this
                should be the directory that contains the images to which the text file refers.
            labels (string or list, optional): `None` or a Python list/tuple. The list/tuple must contain Numpy arrays
                that represent the labels of the dataset.
            image_ids (string or list, optional): `None` or a Python list/tuple . The list/tuple must contain the image
                IDs of the images in the dataset.
            verbose (bool, optional): If `True`, prints out the progress for some constructor operations that may
                take a bit longer.
            show_images (bool, optional) : whether or not to visualize images, can be used to make sure the images used
            during the training are good
        '''

        self.labels_output_format = ('class_id', 'cx', 'cy', 'w', 'h', 'angle')
        self.load_images_into_memory = load_images_into_memory
        self.dataset_size = 0  # As long as we haven't loaded anything yet, the dataset size is zero.
        self.images = None
        self.show_images = show_images

        # The variables `self.filenames`, `self.labels`, and `self.image_ids` below store the output from the parsers.
        # This is the input for the `generate()`` method. `self.filenames` is a list containing all file names of the
        # image samples (full paths).
        # Note that it does not contain the actual image files themselves.
        # `self.labels` is a list containing one 2D Numpy array per image. For an image with `k` ground truth bounding
        # boxes, the respective 2D array has `k` rows` for the respective bounding box.
        # Setting `self.labels` is optional, the generator also works if `self.labels` remains `None`.

        if not filenames is None:
            if isinstance(filenames, (list, tuple)):
                self.filenames = filenames
            elif isinstance(filenames, str):
                with open(filenames, 'rb') as f:
                    self.filenames = [os.path.join(images_dir, line.strip()) for line in f]
            else:
                raise ValueError(
                    "`filenames` must be either a Python list/tuple or a string representing a filepath (to a text file). The value you passed is neither of the two.")
            self.dataset_size = len(self.filenames)
            self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)

            if load_images_into_memory:
                self.images = []
                if verbose:
                    it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
                else:
                    it = self.filenames
                for filename in it:
                    with Image.open(filename) as image:
                        self.images.append(np.array(image, dtype=np.uint8))

        else:
            self.filenames = None

        # In case ground truth is available, `self.labels` is a list containing for each image a list (or NumPy array)
        # of ground truth bounding boxes for that image.
        if not labels is None:
            if isinstance(labels, str):
                with open(labels, 'rb') as f:
                    self.labels = pickle.load(f)
            elif isinstance(labels, (list, tuple)):
                self.labels = labels
            else:
                raise ValueError(
                    "`labels` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.labels = None

        if not image_ids is None:
            if isinstance(image_ids, str):
                with open(image_ids, 'rb') as f:
                    self.image_ids = pickle.load(f)
            elif isinstance(image_ids, (list, tuple)):
                self.image_ids = image_ids
            else:
                raise ValueError(
                    "`image_ids` must be either a Python list/tuple or a string representing the path to a pickled file containing a list/tuple. The value you passed is neither of the two.")
        else:
            self.image_ids = None

    def parse_csv(self,
                  images_dir,
                  labels_filename,
                  include_classes='all',
                  random_sample=False,
                  ret=False,
                  verbose=True):
        '''
        Reads csv file containing the data

        Arguments:
            images_dir (str): The path to the directory that contains the images.
            labels_filename (str): The filepath to a CSV file that contains one ground truth bounding box per line
                and each line contains the following six items: image file name, cx, cy, w, h, angle, class ID.
                The six items have to be in the specific order.
                The class ID is an integer greater than zero. Class ID 0 is reserved for the background class.
                `cx` and `cy` are the coordinates of the center of the box,
                `w` and 'h' are the width and the height of the box,
                'angle' is the angle of the box.
                The image name is expected to be just the name of the image file without the directory path
                at which the image is located. Defaults to `None`.
            include_classes (list, optional): Either 'all' or a list of integers containing the class IDs that
                are to be included in the dataset. Defaults to 'all', in which case all boxes will be included
                in the dataset.
            random_sample (float, optional): Either `False` or a float in `[0,1]`. If this is `False`, the
                full dataset will be used by the generator. If this is a float in `[0,1]`, a randomly sampled
                fraction of the dataset will be used, where `random_sample` is the fraction of the dataset
                to be used. For example, if `random_sample = 0.2`, 20 precent of the dataset will be randomly selected,
                the rest will be ommitted. The fraction refers to the number of images, not to the number
                of boxes, i.e. each image that will be added to the dataset will always be added with all
                of its boxes. Defaults to `False`.
            ret (bool, optional): Whether or not the image filenames and labels are to be returned.
                Defaults to `False`.
            verbose (bool, optional): If `True`, prints out the progress for operations that may take a bit longer.

        Returns:
            None by default, optionally the image filenames and labels.
        '''

        # Set class members.
        self.images_dir = images_dir
        self.labels_filename = labels_filename
        self.include_classes = include_classes

        # Before we begin, make sure that we have a labels_filename
        if self.labels_filename is None:
            raise ValueError("`labels_filename`  have not been set yet. You need to pass it as argument.")

        # Erase data that might have been parsed before
        self.filenames = []
        self.image_ids = []
        self.labels = []

        # First, just read in the CSV file lines and sort them.

        data = []

        input_format = ['image_name', 'cx', 'cy', 'w', 'h', 'angle', 'class_id']

        with open(self.labels_filename, newline='') as csvfile:
            csvread = csv.reader(csvfile, delimiter=',')
            next(csvread)  # Skip the header row.
            for row in csvread:  # For every line (i.e for every bounding box) in the CSV file...
                # If the class_id is among the classes that are to be included in the dataset...
                if self.include_classes == 'all' or int(row[input_format.index('class_id')].strip()) in self.include_classes:
                    box = []  # Store the box class and coordinates here
                    # Select the image name column in the input format and append its content to `box`
                    box.append(row[input_format.index('image_name')].strip())
                    # For each element in the output format (where the elements are the class ID and the four box coordinates)...
                    for element in self.labels_output_format:
                        # ...select the respective column in the input format and append it to `box`.
                        box.append(float(row[input_format.index(element)].strip()))
                    data.append(box)

        data = sorted(data)  # The data needs to be sorted, otherwise the next step won't give the correct result

        # Now that we've made sure that the data is sorted by file names,
        # we can compile the actual samples and labels lists

        # The current image for which we're collecting the ground truth boxes
        current_file = data[0][0]

        # The image ID will be the portion of the image name before the first dot.
        current_image_id = data[0][0].split('.')[0]

        # The list where we collect all ground truth boxes for a given image
        current_labels = []

        for i, box in enumerate(data):

            if box[0] == current_file:  # If this box (i.e. this line of the CSV file) belongs to the current image file
                current_labels.append(box[1:])
                if i == len(data) - 1:  # If this is the last line of the CSV file
                    if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)
                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

            else:  # If this box belongs to a new image file
                if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                    p = np.random.uniform(0, 1)
                    if p >= (1 - random_sample):
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

                else:
                    self.labels.append(np.stack(current_labels, axis=0))
                    self.filenames.append(os.path.join(self.images_dir, current_file))
                    self.image_ids.append(current_image_id)

                current_labels = []  # Reset the labels list because this is a new file.
                current_file = box[0]
                current_image_id = box[0].split('.')[0]
                current_labels.append(box[1:])
                if i == len(data) - 1:  # If this is the last line of the CSV file
                    if random_sample:  # In case we're not using the full dataset, but a random sample of it.
                        p = np.random.uniform(0, 1)
                        if p >= (1 - random_sample):
                            self.labels.append(np.stack(current_labels, axis=0))
                            self.filenames.append(os.path.join(self.images_dir, current_file))
                            self.image_ids.append(current_image_id)

                    else:
                        self.labels.append(np.stack(current_labels, axis=0))
                        self.filenames.append(os.path.join(self.images_dir, current_file))
                        self.image_ids.append(current_image_id)

        self.dataset_size = len(self.filenames)
        self.dataset_indices = np.arange(self.dataset_size, dtype=np.int32)
        if self.load_images_into_memory:
            self.images = []
            if verbose:
                it = tqdm(self.filenames, desc='Loading images into memory', file=sys.stdout)
            else:
                it = self.filenames
            for filename in it:
                with Image.open(filename) as image:
                    self.images.append(np.array(image, dtype=np.uint8))

        if ret:  # In case we want to return these
            return self.filenames, self.labels, self.images


    def generate(self,
                 batch_size=32,
                 shuffle=True,
                 transformations=[],
                 label_encoder=None,
                 returns={'processed_images', 'encoded_labels'},
                 keep_images_without_gt=False,
                 degenerate_box_handling='remove'):
        '''
        Generates batches of samples and (optionally) corresponding labels indefinitely.

        Can shuffle the samples consistently after each complete pass.

        Optionally takes a list of arbitrary image transformations to apply to the
        samples ad hoc.

        Arguments:
            batch_size (int, optional): The size of the batches to be generated.
            shuffle (bool, optional): Whether or not to shuffle the dataset before each pass.
                This option should always be `True` during training, but it can be useful to turn shuffling off
                for debugging or if you're using the generator for prediction.
            transformations (list, optional): A list of transformations that will be applied to the images and labels
                in the given order. Each transformation is a callable that takes as input an image (as a Numpy array)
                and optionally labels (also as a Numpy array) and returns an image and optionally labels in the same
                format.
            label_encoder (callable, optional): Only relevant if labels are given. A callable that takes as input the
                labels of a batch (as a list of Numpy arrays) and returns some structure that represents those labels.
                The general use case for this is to convert labels from their input format to a format that a given object
                detection model needs as its training targets.
            returns (set, optional): A set of strings that determines what outputs the generator yields. The generator's output
                is always a tuple with the processed images as its first element and, if labels and a label encoder are given,
                the encoded labels as its second element. Apart from that, the output tuple can contain additional outputs
                according to the keywords specified here. The possible keyword strings and their respective outputs are:
                * 'processed_images': An array containing the processed images. Will always be in the outputs, so it doesn't
                    matter whether or not you include this keyword in the set.
                * 'encoded_labels': The encoded labels tensor. Will always be in the outputs if a label encoder is given,
                    so it doesn't matter whether or not you include this keyword in the set if you pass a label encoder.
                * 'processed_labels': The processed, but not yet encoded labels. This is a list that contains for each
                    batch image a Numpy array with all ground truth boxes for that image. Only available if ground truth is available.
                * 'filenames': A list containing the file names (full paths) of the images in the batch.
                * 'image_ids': A list containing the integer IDs of the images in the batch. Only available if there
                    are image IDs available.
                * 'original_images': A list containing the original images in the batch before any processing.
                * 'original_labels': A list containing the original ground truth boxes for the images in this batch before any
                    processing. Only available if ground truth is available.
                The order of the outputs in the tuple is the order of the list above. If `returns` contains a keyword for an
                output that is unavailable, that output omitted in the yielded tuples and a warning will be raised.
            keep_images_without_gt (bool, optional): If `False`, images for which there aren't any ground truth boxes before
                any transformations have been applied will be removed from the batch. If `True`, such images will be kept
                in the batch.
            degenerate_box_handling (str, optional): How to handle degenerate boxes, which are boxes that have `w <= 0` and/or
                `h <= 0`. Degenerate boxes can sometimes be in the dataset, or non-degenerate boxes can become degenerate
                after they were processed by transformations. Note that the generator checks for degenerate boxes after all
                transformations have been applied (if any), but before the labels were passed to the `label_encoder` (if one was given).
                Can be one of 'warn' or 'remove'. If 'warn', the generator will merely print a warning to let you know that there
                are degenerate boxes in a batch. If 'remove', the generator will remove degenerate boxes from the batch silently.

        Yields:
            The next batch as a tuple of items as defined by the `returns` argument. By default, this will be
            a 2-tuple containing the processed batch images as its first element and the encoded ground truth boxes
            tensor as its second element if in training mode, or a 1-tuple containing only the processed batch images if
            not in training mode. Any additional outputs must be specified in the `returns` argument.
        '''


        if self.dataset_size == 0:
            raise DatasetError("Cannot generate batches because you did not load a dataset.")

        #############################################################################################
        # Warn if any of the set returns aren't possible.
        #############################################################################################

        if self.labels is None:
            if any([ret in returns for ret in ['original_labels', 'processed_labels', 'encoded_labels']]):
                warnings.warn("Since no labels were given, none of 'original_labels', 'processed_labels' and 'encoded_labels' " +
                              "are possible returns, but you set `returns = {}`. The impossible returns will be missing from the output".format(returns))
        elif label_encoder is None:
            if any([ret in returns for ret in ['encoded_labels']]):
                warnings.warn("Since no label encoder was given, 'encoded_labels' aren't possible returns, " +
                              "but you set `returns = {}`. The impossible returns will be missing from the output".format(returns))
        if (self.image_ids is None) and ('image_ids' in returns):
            warnings.warn("No image IDs were given, therefore 'image_ids' is not a possible return, " +
                          "but you set `returns = {}`. The impossible returns will be missing from the output".format(returns))

        #############################################################################################
        # Do a few preparatory things like maybe shuffling the dataset initially.
        #############################################################################################

        if shuffle:
            objects_to_shuffle = [self.dataset_indices]
            if not (self.filenames is None):
                objects_to_shuffle.append(self.filenames)
            if not (self.labels is None):
                objects_to_shuffle.append(self.labels)
            if not (self.image_ids is None):
                objects_to_shuffle.append(self.image_ids)
            if not (self.images is None):
                objects_to_shuffle.append(self.images)
            shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
            for i in range(len(objects_to_shuffle)):
                objects_to_shuffle[i][:] = shuffled_objects[i]


        if degenerate_box_handling == 'remove':
            box_filter = BoxFilter(check_overlap=False,
                                   check_min_area=False,
                                   check_degenerate=True)

        #############################################################################################
        # Generate mini batches.
        #############################################################################################

        current = 0

        while True:

            batch_X, batch_y = [], []

            if current >= self.dataset_size:
                current = 0

            #########################################################################################
            # Maybe shuffle the dataset if a full pass over the dataset has finished.
            #########################################################################################

                if shuffle:
                    objects_to_shuffle = [self.dataset_indices]
                    if not (self.filenames is None):
                        objects_to_shuffle.append(self.filenames)
                    if not (self.labels is None):
                        objects_to_shuffle.append(self.labels)
                    if not (self.image_ids is None):
                        objects_to_shuffle.append(self.image_ids)
                    if not (self.images is None):
                        objects_to_shuffle.append(self.images)
                    shuffled_objects = sklearn.utils.shuffle(*objects_to_shuffle)
                    for i in range(len(objects_to_shuffle)):
                        objects_to_shuffle[i][:] = shuffled_objects[i]


            #########################################################################################
            # Get the images, image filenames, (maybe) image IDs, and (maybe) labels for this batch.
            #########################################################################################

            # We prioritize our options in the following order:
            # 1) If we have the images already loaded in memory, get them from there.
            # 2) Else, we'll have to load the individual image files from disk.
            batch_indices = self.dataset_indices[current:current + batch_size]
            if not (self.images is None):
                for i in batch_indices:
                    batch_X.append(self.images[i])
                if not (self.filenames is None):
                    batch_filenames = self.filenames[current:current + batch_size]
                else:
                    batch_filenames = None
            else:
                batch_filenames = self.filenames[current:current + batch_size]
                for filename in batch_filenames:
                    with Image.open(filename) as image:
                        batch_X.append(np.array(image, dtype=np.uint8))

            # Get the labels for this batch (if there are any).
            if not (self.labels is None):
                batch_y = deepcopy(self.labels[current:current+batch_size])
            else:
                batch_y = None

            # Get the image IDs for this batch (if there are any).
            if not (self.image_ids is None):
                batch_image_ids = self.image_ids[current:current+batch_size]
            else:
                batch_image_ids = None

            if 'original_images' in returns:
                batch_original_images = deepcopy(batch_X) # The original, unaltered images
            if 'original_labels' in returns and not self.labels is None:
                batch_original_labels = deepcopy(batch_y) # The original, unaltered labels

            current += batch_size

            #########################################################################################
            # Maybe perform image transformations.
            #########################################################################################

            batch_items_to_remove = [] # In case we need to remove any images from the batch, store their indices in this list.
            for i in range(len(batch_X)):

                if not (self.labels is None):
                    # Convert the labels for this image to an array (in case they aren't already).
                    batch_y[i] = np.array(batch_y[i])
                    # If this image has no ground truth boxes, maybe we don't want to keep it in the batch.
                    if (batch_y[i].size == 0) and not keep_images_without_gt:
                        batch_items_to_remove.append(i)
                        continue

                # Apply any image transformations we may have received.
                if transformations:

                    for transform in transformations:

                        if not (self.labels is None):

                            batch_X[i], batch_y[i] = transform(batch_X[i], batch_y[i])

                            if batch_X[i] is None: # In case the transform failed to produce an output image, which is possible for some random transforms.
                                batch_items_to_remove.append(i)
                                continue

                        else:

                            batch_X[i] = transform(batch_X[i])

                if self.show_images:
                    visualize(batch_X[i], gt_labels=batch_y[i])
                #########################################################################################
                # Check for degenerate boxes in this batch item.
                #########################################################################################

                w = 3
                h = 4

                # Check for degenerate ground truth bounding boxes before attempting any computations.
                if np.any(batch_y[i][:,w] <= 0) or np.any(batch_y[i][:,h] <= 0):
                    if degenerate_box_handling == 'warn':
                        warnings.warn("Detected degenerate ground truth bounding boxes for batch item {} with bounding boxes {}, ".format(i, batch_y[i]) +
                                      "i.e. bounding boxes where h <= 0 and/or w <= 0. " +
                                      "This could mean that your dataset contains degenerate ground truth boxes, or that any image transformations you may apply might " +
                                      "result in degenerate ground truth boxes, or that you are parsing the ground truth in the wrong coordinate format." +
                                      "Degenerate ground truth bounding boxes may lead to NaN errors during the training.")
                    elif degenerate_box_handling == 'remove':
                        batch_y[i] = box_filter(batch_y[i])
                        if (batch_y[i].size == 0) and not keep_images_without_gt:
                            batch_items_to_remove.append(i)

            #########################################################################################
            # Remove any items we might not want to keep from the batch.
            #########################################################################################

            if batch_items_to_remove:
                for j in sorted(batch_items_to_remove, reverse=True):
                    # This isn't efficient, but it hopefully shouldn't need to be done often anyway.
                    batch_X.pop(j)
                    batch_filenames.pop(j)
                    if not (self.labels is None): batch_y.pop(j)
                    if not (self.image_ids is None): batch_image_ids.pop(j)
                    if 'original_images' in returns: batch_original_images.pop(j)
                    if 'original_labels' in returns and not (self.labels is None): batch_original_labels.pop(j)

            #########################################################################################

            # CAUTION: Converting `batch_X` into an array will result in an empty batch if the images have varying sizes
            #          or varying numbers of channels. At this point, all images must have the same size and the same
            #          number of channels.

            batch_X = np.array(batch_X)
            if (batch_X.size == 0):
                raise DegenerateBatchError("You produced an empty batch. This might be because the images in the batch vary " +
                                           "in their size and/or number of channels. Note that after all transformations " +
                                           "(if any were given) have been applied to all images in the batch, all images " +
                                           "must be homogenous in size along all axes.")
            ########################################################################################
            # visualize the batch items if needed
            ########################################################################################

            if self.show_images:
                for i in range(len(batch_X)):
                    visualize(batch_X[i], gt_labels=batch_y[i])

            #########################################################################################
            # If we have a label encoder, encode our labels.
            #########################################################################################

            if not (label_encoder is None or self.labels is None):
                batch_y_encoded = label_encoder(batch_y, diagnostics=False)

            else:
                batch_y_encoded = None

            #########################################################################################
            # Compose the output.
            #########################################################################################

            ret = []
            if 'processed_images' in returns: ret.append(batch_X)
            if 'encoded_labels' in returns: ret.append(batch_y_encoded)
            if 'processed_labels' in returns: ret.append(batch_y)
            if 'filenames' in returns: ret.append(batch_filenames)
            if 'image_ids' in returns: ret.append(batch_image_ids)
            if 'original_images' in returns: ret.append(batch_original_images)
            if 'original_labels' in returns: ret.append(batch_original_labels)
            yield tuple(ret)


    def get_dataset(self):
        '''
        Returns:
            3-tuple containing lists and/or `None` for the filenames, labels, image IDs
        '''
        return self.filenames, self.labels, self.image_ids


    def get_dataset_size(self):
        '''
        Returns:
            The number of images in the dataset.
        '''
        return len(self.filenames)
