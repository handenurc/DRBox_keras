'''
Includes:
* Functions to decode and filter raw DRBox model output.
* Functions to perform greedy non-maximum suppression


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

from bounding_box_utils.bounding_box_utils import iou

def _greedy_nms(predictions, iou_threshold=0.45):
    '''
    Perform greedy non-maximum suppression on the input boxes.

    Greedy NMS works by selecting the box with the highest score and
    removing all boxes around it that are too close to it measured by IoU-similarity.
    Out of the boxes that are left over, once again the one with the highest
    score is selected and so on, until no boxes with too much overlap are left.

    Arguments:
        predictions (list): A batch of decoded predictions. For a given batch size `n` this
            is a list of length `n` where each list element is a 2D Numpy array.
            For a batch item with `k` predicted boxes this 2D Numpy array has
            shape `(k, 7)`, where each row contains the coordinates of the respective
            box in the format `[class_id, score, cx, cy, w, h, angle]`.
            Technically, the number of columns doesn't have to be 7, it can be
            arbitrary as long as the first five elements of each row are
            `cx`, `cy`, `w`, `h`, angle (in this order) and the last element
            is the score assigned to the prediction. Note that this function is
            agnostic to the scale of the score or what it represents.
        iou_threshold (float, optional): All boxes with a Jaccard similarity of
            greater than `iou_threshold` with a locally maximal box will be removed
            from the set of predictions, where 'maximal' refers to the box score.

    Returns:
        The predictions after removing non-maxima. The format is the same as the input format.
    '''

    boxes_left = np.copy(predictions)
    maxima = [] # This is where we store the boxes that make it through the non-maximum suppression
    while boxes_left.shape[0] > 0: # While there are still boxes left to compare...
        maximum_index = np.argmax(boxes_left[:,0]) # ...get the index of the next box with the highest confidence...
        maximum_box = np.copy(boxes_left[maximum_index]) # ...copy that box and...
        maxima.append(maximum_box) # ...append it to `maxima` because we'll definitely keep it
        boxes_left = np.delete(boxes_left, maximum_index, axis=0) # Now remove the maximum box from `boxes_left`
        if boxes_left.shape[0] == 0: break # If there are no boxes left after this step, break. Otherwise...

        similarities = iou(boxes_left[:,1:], maximum_box[1:]) # ...compare (IoU) the other left over boxes to the maximum box...
        similarities = np.reshape(similarities, (-1,))
        boxes_left = boxes_left[similarities <= iou_threshold] # ...so that we can remove the ones that overlap too much with the maximum box
        
    return np.array(maxima)

def decode_detections(y_pred,
                      confidence_thresh=0.01,
                      iou_threshold=0.45,
                      top_k=10,
                      normalize_coords=True,
                      img_height=None,
                      img_width=None):
    '''
    Convert model prediction output back to a format that contains only the positive box predictions
    (i.e. the same format that `DRBoxInputEncoder` takes as input).

    After the decoding, two stages of prediction filtering are performed for each class individually:
    First confidence thresholding, then greedy non-maximum suppression. The filtering results for all
    classes are concatenated and the `top_k` overall highest confidence results constitute the final
    predictions for a given batch item.

    Arguments:
        y_pred (array): The prediction output of the DRBox model, expected to be a Numpy array
            of shape `(batch_size, #boxes, #classes + 5 + 5 + 5)`, where `#boxes` is the total number of
            boxes predicted by the model per image and the last axis contains
            `[one-hot vector for the classes, 5 predicted coordinate offsets, 5 anchor box coordinates, 5 variances]`.
        confidence_thresh (float, optional): A float in [0,1), the minimum classification confidence in a specific
            positive class in order to be considered for the non-maximum suppression stage for the respective class.
            A lower value will result in a larger part of the selection process being done by the non-maximum suppression
            stage, while a larger value will result in a larger part of the selection process happening in the confidence
            thresholding stage. Defaults to 0.01, following the paper of SSD.
        iou_threshold (float, optional): A float in [0,1]. All boxes with a Jaccard similarity of greater than `iou_threshold`
            with a locally maximal box will be removed from the set of predictions for a given class, where 'maximal' refers
            to the box score. Defaults to 0.45 following the paper os SSD.
        top_k (int, optional): The number of highest scoring predictions to be kept for each batch item after the
            non-maximum suppression stage.
        normalize_coords (bool, optional): Set to `True` if the model outputs relative coordinates (i.e. coordinates in [0,1])
            and you wish to transform these relative coordinates back to absolute coordinates. If the model outputs
            relative coordinates, but you do not want to convert them back to absolute coordinates, set this to `False`.
            Do not set this to `True` if the model already outputs absolute coordinates, as that would result in incorrect
            coordinates. Requires `img_height` and `img_width` if set to `True`. Defaults to `False`.
        img_height (int, optional): The height of the input images. Only needed if `normalize_coords` is `True`.
        img_width (int, optional): The width of the input images. Only needed if `normalize_coords` is `True`.

    Returns:
        A python list of length `batch_size` where each list element represents the predicted boxes
        for one image and contains a Numpy array of shape `(boxes, 7)` where each row is a box prediction for
        a non-background class for the respective image in the format `[class_id, confidence, cx, cy, w, h, angle]`.
    '''
    if normalize_coords and ((img_height is None) or (img_width is None)):
        raise ValueError("If relative box coordinates are supposed to be converted to absolute coordinates, the decoder needs the image size in order to decode the predictions, but `img_height == {}` and `img_width == {}`".format(img_height, img_width))

    # 1: Convert the box coordinates from the predicted anchor box offsets to predicted absolute coordinates
    # shape of arays is modified to take into account angle
    y_pred_decoded_raw = np.copy(y_pred[:,:,:-10]) # Slice out the classes and the four offsets, throw away the anchor coordinates and variances, resulting in a tensor of shape `[batch, n_boxes, n_classes + 5 coordinates]`

    y_pred_decoded_raw[:,:,[-3,-2]] = np.exp(y_pred_decoded_raw[:,:,[-3,-2]] * y_pred[:,:,[-3,-2]]) # exp(ln(w(pred)/w(anchor)) / w_variance * w_variance) == w(pred) / w(anchor), exp(ln(h(pred)/h(anchor)) / h_variance * h_variance) == h(pred) / h(anchor)
    y_pred_decoded_raw[:,:,[-3,-2]] *= y_pred[:,:,[-8,-7]] # (w(pred) / w(anchor)) * w(anchor) == w(pred), (h(pred) / h(anchor)) * h(anchor) == h(pred)
    
    y_pred_decoded_raw[:,:,[-5,-4]] *= y_pred[:,:,[-5,-4]] * y_pred[:,:,[-8,-7]] # (delta_cx(pred) / w(anchor) / cx_variance) * cx_variance * w(anchor) == delta_cx(pred), (delta_cy(pred) / h(anchor) / cy_variance) * cy_variance * h(anchor) == delta_cy(pred)
    
    y_pred_decoded_raw[:,:,[-5,-4]] += y_pred[:,:,[-10,-9]] # delta_cx(pred) + cx(anchor) == cx(pred), delta_cy(pred) + cy(anchor) == cy(pred)
    y_pred_decoded_raw[:,:,-1] = np.arctan(y_pred_decoded_raw[:,:,-1]) #arctan(tan(angle(pred) - angle(anchor)))
    y_pred_decoded_raw[:,:,-1] += y_pred[:,:,-6] # angle(pred) - angle(anchor) + angle(anchor)

    # 2: If the model predicts normalized box coordinates and they are supposed to be converted back to absolute coordinates, do that
    if normalize_coords:
        y_pred_decoded_raw[:,:,-5] *= img_width # Convert cx back to absolute coordinates
        y_pred_decoded_raw[:,:,-4] *= img_height # Convert cy back to absolute coordinates

        y_pred_decoded_raw[:,:,-3] *= img_width*(np.cos(y_pred_decoded_raw[:,:,-1])**2) + img_height*(np.sin(y_pred_decoded_raw[:,:,-1])**2) # Convert w back to absolute coordinates
        y_pred_decoded_raw[:,:,-2] *= img_height*(np.cos(y_pred_decoded_raw[:,:,-1])**2) + img_width*(np.sin(y_pred_decoded_raw[:,:,-1]) **2) # Convert h back to absolute coordinates


    # 3: Apply confidence thresholding and non-maximum suppression per class
    n_classes = y_pred_decoded_raw.shape[-1] - 5 # The number of classes is the length of the last axis minus the five box coordinates

    y_pred_decoded = [] # Store the final predictions in this list
    for batch_item in y_pred_decoded_raw: # `batch_item` has shape `[n_boxes, n_classes + 5 coords]`
        pred = [] # Store the final predictions for this batch item here
        for class_id in range(1, n_classes): # For each class except the background class (which has class ID 0)...
            single_class = batch_item[:,[class_id, -5, -4, -3, -2, -1]] # ...keep only the confidences for that class, making this an array of shape `[n_boxes, 6]` and...
            threshold_met = single_class[single_class[:,0] > confidence_thresh] # ...keep only those boxes with a confidence above the set threshold.
            
            if threshold_met.shape[0] > 0: # If any boxes made the threshold...
                maxima = _greedy_nms(threshold_met, iou_threshold=iou_threshold) # ...perform NMS on them.
                maxima_output = np.zeros((maxima.shape[0], maxima.shape[1] + 1)) # Expand the last dimension by one element to have room for the class ID. This is now an arrray of shape `[n_boxes, 7]`
                maxima_output[:,0] = class_id # Write the class ID to the first column...
                maxima_output[:,1:] = maxima # ...and write the maxima to the other columns...
                pred.append(maxima_output) # ...and append the maxima for this class to the list of maxima for this batch item.
        
        # Once we're through with all classes, keep only the `top_k` maxima with the highest scores
        if pred: # If there are any predictions left after confidence-thresholding...
            pred = np.concatenate(pred, axis=0)
            if top_k != 'all' and pred.shape[0] > top_k: # If we have more than `top_k` results left at this point, otherwise there is nothing to filter,...
                top_k_indices = np.argpartition(pred[:,1], kth=pred.shape[0]-top_k, axis=0)[pred.shape[0]-top_k:] # ...get the indices of the `top_k` highest-score maxima...
                pred = pred[top_k_indices] # ...and keep only those entries of `pred`...
        else:
            pred = np.array(pred) # Even if empty, `pred` must become a Numpy array.
        y_pred_decoded.append(pred) # ...and now that we're done, append the array of final predictions for this batch item to the output list

    return y_pred_decoded
