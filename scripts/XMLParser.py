import xml.etree.ElementTree as ET
import cv2
import numpy as np
import glob

import pdb

class XMLParser:

    def __init__(self, labelname, img=None, xmlroot=None):

        self._labelname = labelname
        self._img = img
        self._xmlroot = xmlroot

    def writeMaskedImage(self, dest_img):

        cv2.imwrite(dest_img, self._img*self._mask_label)

    def getLabelMask(self, img, xmlroot):

        self._img = img
        self._xmlroot = xmlroot
        self._mask_label = np.zeros((self._img.shape[0], self._img.shape[1]))

        for objectElement in self._xmlroot.findall('object'):

            classnameElement = objectElement.find('name').text

            xElementList = objectElement.findall('polygon/pt/x')
            yElementList = objectElement.findall('polygon/pt/y')

            points = np.zeros((len(xElementList), 2), dtype='int32')
            points_idx = 0
            for (xElement,yElement) in zip(xElementList, yElementList):        
                points[points_idx] = np.array([int(xElement.text), int(yElement.text)])
                points_idx = points_idx + 1

            mask = np.zeros((self._img.shape[0], self._img.shape[1]))
            cv2.fillConvexPoly(mask, points, 1)

            mask = mask.astype(bool)

            if (classnameElement ==  self._labelname): 
                self._mask_label = self._mask_label.astype(bool)
                self._mask_label = mask + self._mask_label

        self._mask_label = np.repeat(self._mask_label[:, :, None], 3, axis=2)

        return self._mask_label

if __name__ == '__main__':
    pass