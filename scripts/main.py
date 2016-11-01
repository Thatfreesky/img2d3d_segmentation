import xml.etree.ElementTree as ET
import cv2
import numpy as np
import glob

import pdb

from CommonImageOperations import CommonImageOperations
from XMLParser import XMLParser
from DisparityComputer import DisparityComputer

def collectFilePaths(src_dir, file_pattern):
    files = glob.glob(src_dir + '/*' + file_pattern)
    files.sort()
    return files

def manuallySegmentDisparities():
    
    # Define Source Directories
    src_dir_anno = '../data/img/terra/405late_20161011194413_3_116_lb'
    src_dir_left = '/media/paloma/Data1/Linux_Data/TERRA/texas_field_tests/20161011/CS_405late_2016-10-11-19-44-13_PIF3_116_lb/qc_l_tr/rectified'
    src_dir_right = '/media/paloma/Data1/Linux_Data/TERRA/texas_field_tests/20161011/CS_405late_2016-10-11-19-44-13_PIF3_116_lb/qc_r_tl/rectified'

    # Read Source File Paths into alist 
    src_xmlfiles = collectFilePaths(src_dir_anno, '.xml')
    src_imgfiles = collectFilePaths(src_dir_anno, '.jpg')
    src_imgfiles_left = collectFilePaths(src_dir_left, '.jpg')
    src_imgfiles_right = collectFilePaths(src_dir_right, '.jpg')

    # Source Image Checks
    assert (len(src_xmlfiles) == len(src_imgfiles)), "number of image and annotation files should be equal"    
    assert (len(src_imgfiles_left) == len(src_imgfiles_right)), "number of left and right images should be equal"

    # Objects and Classes being called
    stemXMLParser = XMLParser('stem')
    dispComputer = DisparityComputer()
    comImgOps = CommonImageOperations()

    # Define Destination Directories
    dest_img_left = '/home/paloma/code/OpenCVReprojectImageToPointCloud/CS_405late_2016-10-11-19-44-13_PIF3_116_lb/rgb-image-'
    dest_disp = '/home/paloma/code/OpenCVReprojectImageToPointCloud/CS_405late_2016-10-11-19-44-13_PIF3_116_lb/disparity-image-'

    file_idx = 0
    for (xmlfile, imgfile, imgfile_right) in zip(src_xmlfiles, src_imgfiles, src_imgfiles_right):

        print 'File Idx : ' + str(file_idx)

        xmlroot = (ET.parse(xmlfile)).getroot()
        img = cv2.imread(imgfile)
        img_left = cv2.imread(imgfile, cv2.CV_LOAD_IMAGE_GRAYSCALE)
        img_right = cv2.imread(imgfile_right, cv2.CV_LOAD_IMAGE_GRAYSCALE)            

        mask_stem = stemXMLParser.getLabelMask(img, xmlroot)
        (disp_left, disp_left_fgnd, img_fgnd) = dispComputer.getDisparity(img_left, img_right)

        img_left = comImgOps.cropImage(img_left, numrows_crop=45, numcols_crop=35)
        disp_left = comImgOps.cropImage(disp_left, numrows_crop=45, numcols_crop=35)
        mask_stem = comImgOps.cropImage(mask_stem, numrows_crop=45, numcols_crop=35)
        
        cv2.imwrite(dest_img_left+str(file_idx)+'.ppm', img_left*mask_stem[:,:,1])
        cv2.imwrite(dest_disp+str(file_idx)+'.pgm', disp_left*mask_stem[:,:,1])

        file_idx = file_idx + 1

def manuallySegmentImages():
    pass

if __name__ == '__main__':
    manuallySegmentDisparities()