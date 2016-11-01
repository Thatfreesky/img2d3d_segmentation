import xml.etree.ElementTree as ET
import cv2
import numpy as np
import re
import glob

import pdb

def collectFilePaths(src_dir, file_pattern):
    files = glob.glob(src_dir + '/*' + file_pattern)
    files.sort()
    return files

def processImage(img):
    (rows, cols, ch) = img.shape
    min_row = 45; min_col = 35; max_row = rows-min_row; max_col = cols-min_col 
    img = img[min_row:max_row, min_col:max_col, :]
    return img

def getLabelMask(xml_root, img_src, classname):
    mask_stems = np.zeros((img_src.shape[0], img_src.shape[1]))

    for objectElement in xml_root.findall('object'):

        classnameElement = objectElement.find('name').text

        xElementList = objectElement.findall('polygon/pt/x')
        yElementList = objectElement.findall('polygon/pt/y')

        points = np.zeros((len(xElementList), 2), dtype='int32')
        points_idx = 0
        for (xElement,yElement) in zip(xElementList, yElementList):        
            points[points_idx] = np.array([int(xElement.text), int(yElement.text)])
            points_idx = points_idx + 1

        mask = np.zeros((img_src.shape[0], img_src.shape[1]))
        cv2.fillConvexPoly(mask, points, 1)

        mask = mask.astype(bool)

        if (classnameElement == classname): 
            mask_stems = mask_stems.astype(bool)
            mask_stems = mask + mask_stems

    mask_stems = np.repeat(mask_stems[:, :, None], 3, axis=2)
    return mask_stems

def main():

    src_dir = '../data/img_src/terra/405late_20161011194413_3_116_lb'
    dest_dir = '../data/img_stems/terra/405late_20161011194413_3_116_lb'
    src_xmlfiles = collectFilePaths(src_dir, '.xml')
    src_imgfiles = collectFilePaths(src_dir, '.jpg')

    assert (len(src_xmlfiles) == len(src_imgfiles)), "number of image and annotation files should be equal"

    file_idx = 0
    for (xmlfile, imgfile) in zip(src_xmlfiles, src_imgfiles):

        xml_root = (ET.parse(xmlfile)).getroot()
        img_src = cv2.imread(imgfile)

        mask_stems = getLabelMask(xml_root, img_src, 'stem')
        img_stems = processImage(mask_stems*img_src)

        filename = imgfile.split('/')[-1]
        cv2.imwrite(dest_dir+'/'+filename, img_stems)

        file_idx = file_idx + 1

if __name__ == '__main__':
    main()