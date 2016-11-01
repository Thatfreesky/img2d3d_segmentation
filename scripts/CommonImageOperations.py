import cv2
import numpy as np

import pdb

class CommonImageOperations:

    def __init__(self):
        pass

    def cropImage(self, img, numrows_crop=45, numcols_crop=35):

        rows = img.shape[0]; cols = img.shape[1] 
        min_row = numrows_crop; min_col = numcols_crop; max_row = rows-min_row; max_col = cols-min_col 
        img = img[min_row:max_row, min_col:max_col, :] if (len(img.shape)==3) else img[min_row:max_row, min_col:max_col]
        return img

if __name__ == '__main__':
    pass