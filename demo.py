# -*- coding: utf-8 -*-
"""
Created on Sat Jan 12 19:04:43 2019

@author: USER
"""

import matplotlib.pyplot as plt

from skimage.feature import greycomatrix, greycoprops
from skimage import data
import numpy as np
import cv2
import pandas
import pylab as pl
from sklearn.feature_extraction import image
import os
import sys
import random
import warnings

from scipy import ndimage as ndi
from skimage import data
from skimage.util import img_as_float
from skimage.filters import gabor_kernel
from skimage.io import imread, imshow, concatenate_images
from numpy import linalg as LA
import csv


img = cv2.imread('C:\\Users\\USER\\Desktop\\salt\\data\\images\\0ba541766e.png')
img1 = cv2.imread('C:\\Users\\USER\\Desktop\\salt\\data\\masks\\0ba541766e.png')

gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray1=cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
patches = image.extract_patches_2d(gray, (2, 2))  
patches1 = image.extract_patches_2d(gray1, (2, 2))

with open('C:\\Users\\USER\\Desktop\\salt\\code\\arsha.csv', 'w') as csvFile:
    for i in range(10):
        gCoMat = greycomatrix(patches[i], [1], [0],256,symmetric=True, normed=True)# Co-occurance matrix
        contrast = greycoprops(gCoMat, prop='contrast')
        dissimilarity = greycoprops(gCoMat, prop='dissimilarity')
        homogeneity = greycoprops(gCoMat, prop='homogeneity')
        energy = greycoprops(gCoMat, prop='energy')
        f=[contrast[0][0],dissimilarity[0][0],energy[0][0]]
        writer=csv.writer(csvFile, delimiter=',')
        writer.writerow(f)
        