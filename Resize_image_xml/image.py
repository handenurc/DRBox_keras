import os
import cv2
import numpy as np
from utils import get_file_name
import xml.etree.ElementTree as ET
from math import cos
from math import sin


def process_image(file_path, output_path, x, y, save_box_images):
    (base_dir, file_name, ext) = get_file_name(file_path)
    image_path = '{}/{}.{}'.format(base_dir, file_name, ext)
    xml = '{}/{}.xml'.format(base_dir, file_name)
    try:
        resize(
            image_path,
            xml,
            (x, y),
            output_path,
            save_box_images=save_box_images,
        )
    except Exception as e:
        print('[ERROR] error with {}\n file: {}'.format(image_path, e))
        print('--------------------------------------------------')


def draw_box(boxes, image, path):
    for i in range(0, len(boxes)):
        cv2.rectangle(image, (boxes[i][2], boxes[i][3]), (boxes[i][4], boxes[i][5]), (0, 255, 0), 1)
    cv2.imwrite(path, image)


def resize(image_path,
           xml_path,
           newSize,
           output_path,
           save_box_images=False,
           verbose=False):

    image = cv2.imread(image_path)

    scale_x = newSize[0] / image.shape[1]
    scale_y = newSize[1] / image.shape[0]

    image = cv2.resize(image, (newSize[0], newSize[1]))

    newBoxes = []
    xmlRoot = ET.parse(xml_path).getroot()
    xmlRoot.find('filename').text = image_path.split('/')[-1]
    size_node = xmlRoot.find('size')
    size_node.find('width').text = str(newSize[0])
    size_node.find('height').text = str(newSize[1])

    for member in xmlRoot.findall('object'):
        robndbox = member.find('robndbox')

        cx = robndbox.find('cx')
        cy = robndbox.find('cy')
        w = robndbox.find('w')
        h = robndbox.find('h')
        angle = robndbox.find('angle')
        
        cx.text = str(int(np.round(float(cx.text) * scale_x)))
        cy.text = str(int(np.round(float(cy.text) * scale_y)))
        w.text = str(int(np.round(float(w.text) * scale_x)))
        h.text = str(int(np.round(float(h.text) * scale_y)))

        newBoxes.append([
            1,
            0,
            # The calculations for drawing the RotatedBbox on the images seem wrong for now. 
            # Therefore resulting images might be misleading. 
            # Showing boxes on images is optional. 
            # This comment will be deleted if any calculations needed to show boxes on the images.
            int(float(cx.text) - ((float(w.text)/2)*cos(float(angle.text))-((float(h.text)/2)*sin(float(angle.text))))),
            int(float(cy.text) - ((float(h.text)/2)*sin(float(angle.text))+((float(h.text)/2)*cos(float(angle.text))))),
            int(float(cx.text) + ((float(w.text)/2)*cos(float(angle.text))+((float(h.text)/2)*sin(float(angle.text))))),
            int(float(cy.text) + ((float(w.text)/2)*sin(float(angle.text))-((float(h.text)/2)*cos(float(angle.text)))))
            ])

    (_, file_name, ext) = get_file_name(image_path)
    cv2.imwrite(os.path.join(output_path, '.'.join([file_name, ext])), image)

    tree = ET.ElementTree(xmlRoot)
    tree.write('{}/{}.xml'.format(output_path, file_name, ext))
    if int(save_box_images):
        save_path = '{}/boxes_images/boxed_{}'.format(output_path, ''.join([file_name, '.', ext]))
        draw_box(newBoxes, image, save_path)
