'''
Includes:
* Function to compute the IoU, and ARIou180, similarity for rectangular, 2D bounding boxes
* Function for coordinate conversion for rectangular, 2D bounding boxes

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


# removed conversions other than minmax2centroids and centroids2minmax
# removed border pixels : always half
def convert_coordinates_axis_aligned(tensor, start_index, conversion):
    '''
    Convert coordinates for axis-aligned 2D boxes between two coordinate formats.

    Creates a copy of `tensor`, i.e. does not operate in place. Currently there are
    2 supported coordinate formats that can be converted from and to each other:
        1) (xmin, xmax, ymin, ymax) - the 'minmax' format
        2) (cx, cy, w, h) - the 'centroids' format

    Arguments:
        tensor (array): A Numpy nD array containing the four consecutive coordinates
            to be converted somewhere in the last axis.
        start_index (int): The index of the first coordinate in the last axis of `tensor`.
        conversion (str, optional): The conversion direction. Can be 'minmax2centroids',
            'centroids2minmax'.

    Returns:
        A Numpy nD array, a copy of the input tensor with the converted coordinates
        in place of the original coordinates and the unaltered elements of the original
        tensor elsewhere.
    '''

    ind = start_index
    tensor1 = np.copy(tensor).astype(np.float)
    if conversion == 'minmax2centroids':
        tensor1[..., ind] = (tensor[..., ind] + tensor[..., ind + 1]) / 2.0  # Set cx
        tensor1[..., ind + 1] = (tensor[..., ind + 2] + tensor[..., ind + 3]) / 2.0  # Set cy
        tensor1[..., ind + 2] = tensor[..., ind + 1] - tensor[..., ind]  # Set w
        tensor1[..., ind + 3] = tensor[..., ind + 3] - tensor[..., ind + 2]  # Set h
    elif conversion == 'centroids2minmax':
        tensor1[..., ind] = tensor[..., ind] - tensor[..., ind + 2] / 2.0  # Set xmin
        tensor1[..., ind + 1] = tensor[..., ind] + tensor[..., ind + 2] / 2.0  # Set xmax
        tensor1[..., ind + 2] = tensor[..., ind + 1] - tensor[..., ind + 3] / 2.0  # Set ymin
        tensor1[..., ind + 3] = tensor[..., ind + 1] + tensor[..., ind + 3] / 2.0  # Set ymax
    else:
        raise ValueError(
            "Unexpected conversion value. Supported values are 'minmax2centroids', 'centroids2minmax', 'corners2centroids', 'centroids2corners', 'minmax2corners', and 'corners2minmax'.")

    return tensor1


def get_corners(labels):
    '''
    Get corners coordinates for 2D rotated boxes

    Arguments:
        labels (array): A Numpy nD array containing the five consecutive coordinate : (cx, cy, w, h, angle).

    Returns:
        A 4-tuple containing the values for the 4 corners.
        Each corner is a n * 2 values tuple containing the x and y coordinates, n being the number of boxes in labels
    '''

    cx, cy, w, h, angle = 1, 2, 3, 4, 5

    # get center of boxes
    centers = np.array([labels[:, cx], labels[:, cy]])

    # get vertices of boxes
    dxcos = labels[:, w] * np.cos(labels[:, angle]) / 2
    dxsin = labels[:, w] * np.sin(labels[:, angle]) / 2
    dycos = labels[:, h] * np.cos(labels[:, angle]) / 2
    dysin = labels[:, h] * np.sin(labels[:, angle]) / 2

    toplefts = centers + np.array([-dxcos - dysin, -dxsin + dycos])
    toprights = centers + np.array([-dxcos - -dysin, -dxsin + -dycos])
    bottomlefts = centers + np.array([dxcos - dysin, dxsin + dycos])
    bottomrights = centers + np.array([dxcos - -dysin, dxsin + -dycos])

    return toplefts, toprights, bottomlefts, bottomrights


def get_centroids_coords(toplefts, toprights, bottomlefts, bottomrights):
    '''
    Get centroids coordinates for 2D boxes

    Arguments:
        toplefts (tuple) : n * 2 value tuple containing the x and y coordinates of top lefts corners
        toprights (tuple): n * 2 value tuple containing the x and y coordinates of top rights corners
        bottomlefts (tuple): n * 2 value tuple containing the x and y coordinates of bottom lefts corners
        bottomrights (tuple): n * 2 value tuple containing the x and y coordinates of bottom rights corners

    Returns:
        A Numpy nD array containing the coordinates of the boxes in the centroids format
    '''

    cx = np.mean([toplefts[0], toprights[0], bottomlefts[0], bottomrights[0]], axis=0)
    cy = np.mean([toplefts[1], toprights[1], bottomlefts[1], bottomrights[1]], axis=0)

    w = np.sqrt((toplefts[0] - bottomlefts[0]) ** 2 + (toplefts[1] - bottomlefts[1]) ** 2)
    h = np.sqrt((toplefts[0] - toprights[0]) ** 2 + (toplefts[1] - toprights[1]) ** 2)

    angle = np.arctan((toplefts[1] - bottomlefts[1]) / (toplefts[0] - bottomlefts[0]))
    angle = np.mod(angle, np.pi)

    return cx, cy, w, h, angle


def rotate_box(boxes, rect_base):
    '''
    Return rotated rectangles by the angle of rect_base. Only the coordinates cx, cy are rotated.
    We do all the transpose operations in order to save time during batch generation

    Arguments:
        boxes (array): A Numpy nD array containing the boxes to be rotated
        rect_base (array): the box which contains the angle to rotate the boxes in the array boxes

    Returns:
        A Numpy nD array with the same size as the argument boxes, the rotated boxes.
    '''

    # get angle of rect_base
    theta = rect_base[4]

    # transition matrix
    trans = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    boxes = np.copy(boxes.T)
    rot = np.copy(boxes[:2])

    rect_base_coords = rect_base[:2]

    # translate boxes by rect_base coordinates
    rot = (rot.T - rect_base_coords)
    # rotate cx, cy of boxes
    rot = np.dot(rot, trans)

    # translate back again with coordinates of rect_base
    rot = rot + rect_base_coords

    boxes[:2] = rot.T
    boxes = np.swapaxes(boxes, 1, 0)

    return boxes


def intersection_area_training(boxes1, boxes2):
    '''
    Computes the intersection areas (with the formula of ARiou180) of two sets of 2D rectangular boxes.
    Used to compute similarity during training.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    returns an `(m,n)` matrix with the intersection areas for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    The boxes in boxes2 are rotated to have the same angle as the boxes in boxes1 to compute the intersection area of ARiou180.
    We apply a rotation to the centers of boxes2 by the angle of the boxes in boxes1, and then we consider both angles to be zero.
    This way we can use numpy to calculate the intersection area and increase the speed.

    We do all the transpose operations in order to save time during batch generation

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the
            format centroids or a 2D Numpy array of shape `(m, 5)` containing the coordinates for `m` boxes.
        boxes2 (array): Either a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the
            format centroids or a 2D Numpy array of shape `(n, 5)` containing the coordinates for `n` boxes.

    Returns:
        A 2D Numpy array of dtype float containing values with the intersection areas of the boxes in `boxes1` and `boxes2`.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 5):
        raise ValueError(
            "All boxes must consist of 5 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(
                boxes1.shape[1], boxes2.shape[1]))

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    inter_areas = np.zeros((m, n))
    xmin = 0
    xmax = 1
    ymin = 2
    ymax = 3

    # iterate over boxes1
    for i, b1 in enumerate(boxes1):
        # rotate boxes2 with the angle of box b
        rotated_boxes = rotate_box(boxes2, b1)

        # convert coordinates to minmax, we consider that the angles of both boxes are zero
        rotated_boxes = rotated_boxes.T

        rotated_boxes = convert_coordinates_axis_aligned(rotated_boxes[:4].T, 0, 'centroids2minmax')
        b1 = convert_coordinates_axis_aligned(b1[:4], 0, 'centroids2minmax')

        rotated_boxes = rotated_boxes.T
        # get the greater xmin and ymin values.
        min_xy = np.maximum(rotated_boxes[[xmin, ymin]].T, b1[[xmin, ymin]])

        # get the smaller xmax and ymax values.
        max_xy = np.minimum(rotated_boxes[[xmax, ymax]].T, b1[[xmax, ymax]])

        # Compute the side lengths of the intersection rectangles.
        side_lengths = np.maximum(0, max_xy - min_xy)

        side_lengths = side_lengths.T
        inter_areas[i, :] = (side_lengths[0] * side_lengths[1]).T

    return inter_areas


class Vector:
    '''
    Class representing a point
    '''

    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __add__(self, v):
        return Vector(self.x + v.x, self.y + v.y)

    def __sub__(self, v):
        return Vector(self.x - v.x, self.y - v.y)

    def cross(self, v):
        return self.x * v.y - self.y * v.x


class Line:
    '''
    Class representing an edge of a bounding box
    '''

    # ax + by + c = 0
    def __init__(self, v1, v2):
        self.a = v2.y - v1.y
        self.b = v1.x - v2.x
        self.c = v2.cross(v1)

    def __call__(self, p):
        '''
        Computes ax + by + c for a new point p
        Determines on wich side of the line the point is.

        Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        any point p with line(p) > 0 is on the "outside".

        '''
        return self.a * p.x + self.b * p.y + self.c

    def intersection(self, other):
        '''
        Get intersection point between this line and another line
        '''

        w = self.a * other.b - self.b * other.a
        return Vector(
            (self.b * other.c - self.c * other.b) / w,
            (self.c * other.a - self.a * other.c) / w
        )


def rectangle_vertices(cx, cy, w, h, r):
    '''
    Compute the angles of a bounding box and returns objects of the class Vector
    '''

    angle = r
    dx = w / 2
    dy = h / 2
    dxcos = dx * np.cos(angle)
    dxsin = dx * np.sin(angle)
    dycos = dy * np.cos(angle)
    dysin = dy * np.sin(angle)

    return (
        Vector(cx, cy) + Vector(-dxcos - -dysin, -dxsin + -dycos),
        Vector(cx, cy) + Vector(dxcos - -dysin, dxsin + -dycos),
        Vector(cx, cy) + Vector(dxcos - dysin, dxsin + dycos),
        Vector(cx, cy) + Vector(-dxcos - dysin, -dxsin + dycos)
    )


def intersection_area_(r1, r2):
    '''
    
    Computes the real intersection area of two rotated bounding boxes 
    Used during decoding in intersection_area_decoding.

    Arguments:
        r1 (array): a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the format (cx, cy, w, h, angle)
        r2 (array): a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the format (cx, cy, w, h, angle)
    
    Returns:
        a float representing the intersection area of r1 and r2

    '''

    # First convert r1 and r2 into a sequence of vertices
    rect1 = rectangle_vertices(*r1)
    rect2 = rectangle_vertices(*r2)

    # Use the vertices of the first rectangle as
    # starting vertices of the intersection polygon.
    intersection = rect1

    # Loop over the edges of the second rectangle
    for p, q in zip(rect2, rect2[1:] + rect2[:1]):
        if len(intersection) <= 2:
            break  # No intersection

        line = Line(p, q)

        # Any point p with line(p) <= 0 is on the "inside" (or on the boundary),
        # Any point p with line(p) > 0 is on the "outside".

        # Loop over the edges of the intersection polygon,
        # and determine which part is inside and which is outside.
        new_intersection = []
        line_values = [line(t) for t in intersection]
        for s, t, s_value, t_value in zip(
                intersection, intersection[1:] + intersection[:1],
                line_values, line_values[1:] + line_values[:1]):

            if s_value <= 0:
                new_intersection.append(s)
            if s_value * t_value < 0:
                # Points are on opposite sides.
                # Add the intersection of the lines to new_intersection.
                intersection_point = line.intersection(Line(s, t))
                new_intersection.append(intersection_point)

        intersection = new_intersection

    # Calculate area
    if len(intersection) <= 2:
        return 0

    # return intersection area
    return 0.5 * sum(p.x * q.y - p.y * q.x for p, q in
                     zip(intersection, intersection[1:] + intersection[:1]))


def intersection_area_decoding(boxes1, boxes2):
    '''
    Computes the intersection areas of two sets of 2D rectangular boxes.
    The function is used for decoding raw predictions with non-maximum suppression (NMS)

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    returns an `(m,n)` matrix with the intersection areas for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the
            format centroids or a 2D Numpy array of shape `(m, 5)` containing the coordinates for `m` boxes.
        boxes2 (array): Either a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the
            format centroids or a 2D Numpy array of shape `(n, 5)` containing the coordinates for `n` boxes.

    Returns:
        A 2D Numpy array of dtype float containing values with the intersection areas of the boxes in `boxes1` and `boxes2`.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 5):
        raise ValueError(
            "All boxes must consist of 5 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(
                boxes1.shape[1], boxes2.shape[1]))

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    inter_areas = np.zeros((m, n))

    for i, b1 in enumerate(boxes1):
        for j, b2 in enumerate(boxes2):
            inter_areas[i, j] = intersection_area_(b1, b2)

    return inter_areas


def sum_area_(boxes1, boxes2):
    '''
    Computes the sum of areas of two sets of axis-aligned 2D rectangular boxes.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    returns an `(m,n)` matrix with the sum of the areas for all possible
    combinations of the boxes in `boxes1` and `boxes2`.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the
            format centroids or a 2D Numpy array of shape `(m, 5)` containing the coordinates for `m` boxes.
        boxes2 (array): Either a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the
            format centroids or a 2D Numpy array of shape `(n, 5)` containing the coordinates for `n` boxes.

    Returns:
        A 2D Numpy array of dtype float containing values with the sum of the areas of the boxes in `boxes1` and `boxes2`.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 5):
        raise ValueError(
            "All boxes must consist of 5 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(
                boxes1.shape[1], boxes2.shape[1]))

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    areas1 = boxes1[:, 2] * boxes1[:, 3]  # w*h
    areas2 = boxes2[:, 2] * boxes2[:, 3]  # w*h

    s1 = np.tile(np.expand_dims(areas1, axis=1), reps=(1, n))
    s2 = np.tile(np.expand_dims(areas2, axis=0), reps=(m, 1))
    return s1 + s2


def ARiou180(boxes1, boxes2):
    '''
    Computes the modified version of intersection-over-union similarity, ARIou180, of two sets of rotated 2D rectangular boxes. 
    Used only for training.

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    returns an `(m,n)` matrix with the ARIoU180s for all possible combinations of the boxes in `boxes1` and `boxes2`.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the
            format centroids or a 2D Numpy array of shape `(m, 5)` containing the coordinates for `m` boxes.
        boxes2 (array): Either a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the
            format centroids or a 2D Numpy array of shape `(n, 5)` containing the coordinates for `n` boxes.

    Returns:
        A 2D Numpy array of dtype float containing values in [0,1], the ARiou180 similarity of the boxes in `boxes1` and
        `boxes2`.
        0 means there is no overlap between two given boxes, 1 means their coordinates are identical.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 5):
        raise ValueError(
            "All boxes must consist of 5 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(
                boxes1.shape[1], boxes2.shape[1]))

    m = boxes1.shape[0]  # The number of boxes in `boxes1`
    n = boxes2.shape[0]  # The number of boxes in `boxes2`

    # Compute the cosine of the difference in angles for all possible combinations of the boxes in `boxes1` and `boxes2`
    b1 = np.tile(np.expand_dims(boxes1[:, 4], axis=1), reps=(1, n))
    b2 = np.tile(np.expand_dims(boxes2[:, 4], axis=0), reps=(m, 1))
    cos_matrix = np.abs(np.cos(b1 - b2))

    # Compute the intersection areas
    intersection_areas = intersection_area_training(boxes1, boxes2)

    # Compute the union areas.
    sum_areas = sum_area_(boxes1, boxes2)
    union_areas = sum_areas - intersection_areas

    return intersection_areas * cos_matrix / union_areas


def iou(boxes1, boxes2):
    '''
    Computes the intersection-over-union similarity (also known as Jaccard similarity)
    of two sets of rotated 2D rectangular boxes. Used only for decoding raw predictions with non-maximum suppression (NMS).

    Let `boxes1` and `boxes2` contain `m` and `n` boxes, respectively.

    returns an `(m,n)` matrix with the IoUs for all possible combinations of the boxes in `boxes1` and `boxes2`.

    Arguments:
        boxes1 (array): Either a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the
            format centroids or a 2D Numpy array of shape `(m, 5)` containing the coordinates for `m` boxes.
        boxes2 (array): Either a 1D Numpy array of shape `(5, )` containing the coordinates for one box in the
            format centroids or a 2D Numpy array of shape `(n, 5)` containing the coordinates for `n` boxes.

    Returns:
        A 2D Numpy array of dtype float containing values in [0,1], the Jaccard similarity of the boxes in `boxes1` and `
        boxes2`. 
        0 means there is no overlap between two given boxes, 1 means their coordinates are identical.
    '''

    # Make sure the boxes have the right shapes.
    if boxes1.ndim > 2: raise ValueError("boxes1 must have rank either 1 or 2, but has rank {}.".format(boxes1.ndim))
    if boxes2.ndim > 2: raise ValueError("boxes2 must have rank either 1 or 2, but has rank {}.".format(boxes2.ndim))

    if boxes1.ndim == 1: boxes1 = np.expand_dims(boxes1, axis=0)
    if boxes2.ndim == 1: boxes2 = np.expand_dims(boxes2, axis=0)

    if not (boxes1.shape[1] == boxes2.shape[1] == 5):
        raise ValueError(
            "All boxes must consist of 5 coordinates, but the boxes in `boxes1` and `boxes2` have {} and {} coordinates, respectively.".format(
                boxes1.shape[1], boxes2.shape[1]))

    # Compute intersection areas
    intersection_areas = intersection_area_decoding(boxes1, boxes2)

    # Compute the union areas.
    sum_areas = sum_area_(boxes1, boxes2)
    union_areas = sum_areas - intersection_areas

    return intersection_areas / union_areas
