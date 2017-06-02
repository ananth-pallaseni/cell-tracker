# Takes in a list of blobs and draws only those blobs into a 4d stack the same shape as the reference

import tifffile as tf
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

import json
import sys

# Extract the raw image data from an ome-tiff file
# Note: returns a numpy array in the format (time, z-val, y-val, x-val) with dtype=uint8
def loadTiff(fname):
    imraw = tf.TiffFile(fname)
    im = imraw.asarray()
    imraw.close()
    return im



def create_stack(all_blobs, reference, outfile='cut_stack.tif'):

	time, depth, height, width = loadTiff(reference).shape

	channels = 1
	newshape = (time, depth, channels, height, width)

	# Create new stack
	newstack = np.zeros( newshape , dtype='uint8')
	
	# Populate stack with blob pixels
	for t in range(time):
		for blob in all_blobs[t]:
			for z, y, x in blob:
				newstack[t, z, 0, y, x] = 255

	tf.imsave(outfile, newstack, imagej=True)

all_blobs = []
with open('tracked_filtered_blobs.json') as f:
	all_blobs = json.load(f)

ref = 'auto-proc-whole-stack.tif'

create_stack(all_blobs, ref)


