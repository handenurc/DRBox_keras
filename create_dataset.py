'''

Create a csv file containing the data to use during training and evaluation

For example you can use the script to generate data file for cars with the following command :

python create_dataset.py -d data/Car/train_data/ -v 0.1 -t 0.1 -s -f data/Car -o c

Copyright © 2019 THALES ALENIA SPACE FRANCE. All rights reserved

Author : Paul Pontisso
'''

import os
import csv
import numpy as np
import argparse
from sklearn.model_selection import train_test_split

def get_list_of_labels(file):
    '''
    returns a list containing all labels in the file
    '''

    try:
        with open(file, "r") as f:
            rboxes = f.readlines()

    except FileNotFoundError:
        return None

    l = []
    for i, rbox in enumerate(rboxes):
        cx, cy, w, h, label, angle = rbox.split(' ')
        cx, cy, w, h, label, angle = float(cx), float(cy), float(w), float(h), float(label), float(angle)

        angle = 2 * np.pi - angle * np.pi / 180
        img_file = file.split('/')[-1][:-5]

        l.append([img_file, cx, cy, w, h, angle, label])

    return l

if __name__ == '__main__':

    # Command line argument parsing
    parser = argparse.ArgumentParser(description='Script to create dataset for DRBox')

    parser.add_argument('-d', '--data_folder',
                        type=str,
                        help="Data folder to read the image and labels.")

    parser.add_argument('-s', '--shuffle',
                        action='store_true',
                        help="Whether or not to shuffle the images")

    parser.add_argument('-f', '--save_folder',
                        type=str,
                        default='',
                        help="Folder to save the csv files containing the data")

    parser.add_argument('-v', '--validation_split',
                        type=float,
                        default=0,
                        help="The percentage of image to use in validation set.")

    parser.add_argument('-t', '--test_split',
                        type=float,
                        default=0,
                        help="The percentage of image to use in test set.")

    parser.add_argument('-o', '--object',
                        type=str,
                        default='a',
                        required=True,
                        help="The type of objects in the dataset : can be either 'a' for airplane, 'c' for cars, or 's' for ships")

    args = parser.parse_args()

    #                          COMMAND LINE PARAMETERS
    # ______________________________________________________________________________

    data_folder = args.data_folder
    save_folder = args.save_folder
    validation_split = args.validation_split
    test_split = args.test_split
    shuffle = args.shuffle
    object = args.object

    if object not in ['a', 'c', 's']:
        raise ValueError(
            "object type must be either 'a' for airplane, 'c' for cars, or 's' for ships but is : ".format(object))

    labels_train = []
    labels_validate = []
    labels_test = []

    labels_train.append(['frame', 'cx', 'cy', 'w', 'h', 'angle', 'class'])
    labels_validate.append(['frame', 'cx', 'cy', 'w', 'h', 'angle', 'class'])
    labels_test.append(['frame', 'cx', 'cy', 'w', 'h', 'angle', 'class'])

    # fill the labels list with all labels
    labels = []
    nb_images = 0
    for img in os.listdir(data_folder):
        if not img.endswith('.tif'):
            continue

        rbox_file = img + '.rbox'
        l = get_list_of_labels(os.path.join(data_folder, rbox_file))
        if l is not None:
            labels.extend(l)
            nb_images += 1

    print('Number of images                        : {}'.format(nb_images))

    # BAOSHANJICHANG_Level_19_330.tif_res_0.707_874.tif

    # get all the uniques files
    base_names = []

    print(len(labels))
    for l in labels:

        if len(l[0].split('_')) == 4 or object == 'a' or object == 's':

            base_name = l[0].split('.')[0]

            if base_name not in base_names:
                base_names.append(base_name)

    if shuffle:
        base_names_remain, base_names_test = train_test_split(base_names, test_size=test_split)
        base_names_train, base_names_validate = train_test_split(base_names_remain, test_size=validation_split)
    else:
        test_size = int(test_split * len(base_names))
        validation_size = int(validation_split * len(base_names))

        base_names_test = base_names[-test_size:]
        base_names_validate = base_names[-validation_size - test_size: -test_size]
        base_names_train = base_names[:-validation_size - test_size]


    # get labels from files
    for l in labels:

        base_name = l[0].split('.')[0]

        if object == 'c' and len(base_name.split('_')) == 5:
            base_name = '_'.join(base_name.split('_')[:-1])

        if base_name in base_names_train:
            labels_train.append(l)
        elif base_name in base_names_validate:
            labels_validate.append(l)
        elif base_name in base_names_test:
            labels_test.append(l)

    print('Number of objects in the train set      : {}'.format(len(labels_train) - 1))
    print('Number of objects in the validation set : {}'.format(len(labels_validate) - 1))
    print('Number of objects in the test set       : {}'.format(len(labels_test) - 1))

    if len(labels_train) > 1:
        csvfile = os.path.join(save_folder, 'labelstrain.csv')
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(labels_train)

    if len(labels_validate) > 1:
        csvfile = os.path.join(save_folder, 'labelsvalidate.csv')
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(labels_validate)

    if len(labels_test) > 1:
        csvfile = os.path.join(save_folder, 'labelstest.csv')
        with open(csvfile, "w") as output:
            writer = csv.writer(output, lineterminator='\n')
            writer.writerows(labels_test)
