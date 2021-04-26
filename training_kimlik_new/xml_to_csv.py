import os
import glob
import pandas as pd
import xml.etree.ElementTree as ET


def xml_to_csv(path):
    xml_list = []
    for xml_file in glob.glob(path + '/*.xml'):
        tree = ET.parse(xml_file)
        root = tree.getroot()
        for member in root.findall('object'):
            value = (root.find('filename').text,
                    #  int(root.find('size')[0].text),
                    #  int(root.find('size')[1].text),
                     
                     int(member[5][0].text),
                     int(member[5][1].text),
                     int(member[5][2].text),
                     int(member[5][3].text),
                     int(float(member[5][4].text)),
                     # member[1].text, # should be this if more than 1 class is possible
                     1,
                     )
            xml_list.append(value)
    # column_name = ['filename', 'width', 'height', 'class', 'xmin', 'ymin', 'xmax', 'ymax']
    column_name_new = ['frame', 'cx', 'cy', 'w', 'h', 'angle', 'class']
    xml_df = pd.DataFrame(xml_list, columns=column_name_new)
    return xml_df


def main():
    image_path = os.path.join(os.getcwd(), 'training_kimlik_new')
    xml_df = xml_to_csv(image_path)
    xml_df.to_csv('kimlik_labels_sadece_kimlik.csv', index=None)
    print('Successfully converted xml to csv.')


main()
