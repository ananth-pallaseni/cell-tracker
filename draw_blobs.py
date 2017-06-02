# Looks for a json file containing blobs and then draws those blobs 

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

# Finds centroid of a list of pixels
def centroid(blob):
	sumz = 0
	sumy = 0
	sumx = 0
	for z, y, x in blob:
		sumz += z 
		sumy += y 
		sumx += x 
	sumz = sumz / len(blob)
	sumy = sumy / len(blob) 
	sumx = sumx / len(blob)
	return (sumz, sumy, sumx)

# Takes in an array containing arrays of pixels
# Returns an array of bounding boxes in the form (blobid, (minz, maxz, miny, maxy, minx, maxx))
def get_bounding_boxes(blobs):
	bboxes = []

	for i in range(len(blobs)):
	    zs = [b[0] for b in blobs[i]]
	    minz = min(zs)
	    maxz = max(zs)

	    ys = [b[1] for b in blobs[i]]
	    miny = min(ys)
	    maxy = max(ys)

	    xs = [b[2] for b in blobs[i]]
	    minx = min(xs)
	    maxx = max(xs)

	    bboxes.append((minz, maxz, miny, maxy, minx, maxx))

	return bboxes



def drawbox(im, y1, y2, x1, x2):
    cv2.rectangle(im, (x1, y1), (x2, y2), (0, 255, 0), 2)


def draw_no_ref(dims, blobs, outfile):
	time, depth, height, width = dims 
	channels = 1
	rgb = 3
	newshape = (time, depth, channels, height, width, rgb)
	newstack = np.zeros( newshape , dtype='uint8')

	for t in range(len(blobs)):
		# Grab blob dictionary for time t
		blob_slice = blobs[t]

		# Get bounding boxes at time t
		bboxes = get_bounding_boxes(blob_slice)

		# Grab coordinates for boxes in 3d stack
		for (az, bz, ay, by, ax, bx) in bboxes:
			# For each z coordinate
			for z in range(az, bz+1):
				# Draw boxes
				drawbox(newstack[t, z, 0, :, :, :], ay, by, ax, bx)

		for blob in blob_slice:
			for zz, yy, xx in blob:
				newstack[t,zz,0,yy,xx,0] = 255
				newstack[t,zz,0,yy,xx,1] = 0
				newstack[t,zz,0,yy,xx,2] = 0



	print('Saving to ' + outfile)
	tf.imsave(outfile, newstack, imagej=True)
	return newstack

def draw_copy_of(cpy_file, blobs, outfile):
	stack4d = loadTiff(cpy_file)
	draw_no_ref(stack4d.shape, blobs, outfile)




def draw_bboxes(stack4d, blobs, outfile='bounding-boxes-4d.tif'):
	time, depth, height, width = stack4d.shape

	channels = 1
	rgb = 3
	newshape = (time, depth, channels, height, width, rgb)

	# Create new rgb version of stack4d
	newstack = np.zeros( newshape , dtype='uint8')
	for t in range(time):
		sl = stack4d[t]
		newstack[t,:,0,:,:,0] = sl 
		newstack[t,:,0,:,:,1] = sl 
		newstack[t,:,0,:,:,2] = sl 

	for t in range(len(blobs)):
		# Grab blob dictionary for time t
		blob_slice = blobs[t]

		# Get bounding boxes at time t
		bboxes = get_bounding_boxes(blob_slice)

		# Grab coordinates for boxes in 3d stack
		for (az, bz, ay, by, ax, bx) in bboxes:
			# For each z coordinate
			for z in range(az, bz+1):
				# Draw boxes
				drawbox(newstack[t, z, 0, :, :, :], ay, by, ax, bx)

		for blob in blob_slice:
			for zz, yy, xx in blob:
				newstack[t,zz,0,yy,xx,0] = 255
				newstack[t,zz,0,yy,xx,1] = 0
				newstack[t,zz,0,yy,xx,2] = 0



	print('Saving to ' + outfile)
	tf.imsave(outfile, newstack, imagej=True)
	return newstack

# Draw bounding boxes onto image stack
def draw_bboxes_from_file(fname, blobs, outfile='bounding-boxes-4d.tif'):
	# Load 4d stack
	stack4d = loadTiff(fname)

	return draw_bboxes(stack4d, blobs, outfile=outfile)

	


def load_all_blobs(fname):
	all_blobs = []
	with open(fname) as f:
		all_blobs = json.load(f)
	return all_blobs 




if __name__ == '__main__':
	infile = None
	reference = None
	outfile = None
	argc = len(sys.argv)
	if argc > 1:
		infile = sys.argv[1]
	if argc	> 2:
		outfile = sys.argv[2]
	if argc > 3:
		reference = sys.argv[3]

	if infile is not None and outfile is not None and reference is not None:
		blob_stack = load_all_blobs(infile)
		draw_bboxes_from_file(reference, blob_stack, outfile=outfile)