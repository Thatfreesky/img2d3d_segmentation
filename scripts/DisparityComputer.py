import cv2
import numpy as np

import pdb

class DisparityComputer:

    def __init__(self, img_left=None, img_right=None):
        self._img_left = img_left
        self._img_right = img_right

        # StereoSGBM Parameters
        self._max_disparity = 16*6
        self._sad_winsize = 11
        self._uniqueness_ratio = 0
        self._P1 = 1000
        self._P2 = 4000
        self._speckle_winsize = 125
        self._speckle_range = 1

        self._fgnd_disp_thresh = 40

    def leftRightHistEqualize(self, img_left, img_right) :
        
        lookup = np.zeros([256], np.uint8)
        cdf_new = np.zeros([256])
        rows,cols = img_right.shape
        img_left_new = np.zeros([rows,cols], np.uint8)

        hist_left = cv2.calcHist([img_left],[0],None,[256],[0,255])
        hist_right = cv2.calcHist([img_right],[0],None,[256],[0,255])
        cdf_left_norm = hist_left.cumsum() /(rows*cols)
        cdf_right_norm = hist_right.cumsum() /(rows*cols)

        for i in range(0,256):
            lookup[i] = np.argmin(np.abs(cdf_left_norm[i]-cdf_right_norm))
            # cdf_new[i] = cdf_right_norm[lookup[i]]
            img_left_new[ np.where(img_left==i) ] = lookup[i]
        
        return img_left_new

    def imgHistEqualize(self, img, clipLimit=2.0, tileGridSize=(8,8)):
        clahe = cv2.createCLAHE(clipLimit, tileGridSize)
        img = clahe.apply(img)
        return img

    def extractForeground(self, disparity, image, threshold):
     img_foreground = np.zeros((image.shape[0], image.shape[1]), np.uint8)
     disp_foreground = np.zeros((image.shape[0], image.shape[1]), np.uint8)

     for i in range(0, image.shape[0]):
      for j in range(0, image.shape[1]):    
       if disparity[i,j] > threshold:
        img_foreground[i,j] = image[i,j]
        disp_foreground[i,j] = disparity[i,j] 

     return img_foreground, disp_foreground

    def stereoSGBM(self, img_left, img_right):

        stereo = cv2.StereoSGBM(minDisparity=0, numDisparities=self._max_disparity, SADWindowSize=self._sad_winsize, uniquenessRatio=self._uniqueness_ratio, P1=self._P1, P2=self._P2, speckleWindowSize=self._speckle_winsize, speckleRange=self._speckle_range)
        disp_left = stereo.compute(img_left, img_right)
        disp_left_visual = np.zeros((img_left.shape[0], img_left.shape[1]), np.uint8)
        disp_left_visual = cv2.normalize(disp_left, alpha=0, beta=self._max_disparity, norm_type=cv2.cv.CV_MINMAX, dtype=cv2.cv.CV_8U)

        img_foreground, disp_foreground = self.extractForeground(disp_left_visual, img_left, self._fgnd_disp_thresh)

        return (disp_left_visual, disp_foreground, img_foreground)

    def writeImageDisparity(self, dest_img, dest_disp):
        cv2.imwrite(dest_img, self._img_left)
        cv2.imwrite(dest_disp, self._disp_left)

    def getDisparity(self, img_left, img_right):

        self._img_left = img_left
        self._img_right = img_right

        self._img_left = self.leftRightHistEqualize(self._img_left, self._img_right)
        # self._img_left = self.imgHistEqualize(self._img_left)
        # self._img_right = self.imgHistEqualize(self._img_right)

        (self._disp_left, self._disp_left_fgnd, self._img_fgnd) = self.stereoSGBM(self._img_left, self._img_right)

        return (self._disp_left, self._disp_left_fgnd, self._img_fgnd)

if __name__ == '__main__':
    pass

