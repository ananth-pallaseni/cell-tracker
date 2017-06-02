import tifffile as tf
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

import json
import math 
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
	sumz = int(sumz / len(blob))
	sumy = int(sumy / len(blob) )
	sumx = int(sumx / len(blob))
	return (sumz, sumy, sumx)

def box_centroid(box):
	z = int((box[0] + box[1]) / 2)
	y = int((box[2] + box[3]) / 2)
	x = int((box[4] + box[5]) / 2)
	return (z, y, x)


def bounding_box(blob):
	zs = [pix[0] for pix in blob]
	ys = [pix[1] for pix in blob]
	xs = [pix[2] for pix in blob]

	minz = min(zs)
	maxz = max(zs)

	miny = min(ys)
	maxy = max(ys)

	minx = min(xs)
	maxx = max(xs)

	return (minz, maxz, miny, maxy, minx, maxx)

def centroid_adjust(cent, target):
	cz, cy, cx = cent 
	tz, ty, tx = target 
	return tz-cz, ty-cy, tx-cx

def blob_adjust(blob, target):
	cent = centroid(blob)
	dz, dy, dx = centroid_adjust(cent, target)

	new_blob = []
	for z, y, x in blob:
		new_pix = z+dz, y+dy, x+dx
		new_blob.append( new_pix )
	return new_blob

def adjust_track(blobs, target):
	new_blobs = []
	for b in blobs:
		new_blobs.append(blob_adjust(b, target))
	return new_blobs



track = []
with open('individual_track0.json') as f:
	track = json.load(f)

# Format of track is: track[i] is list of blobs. A blob is a list of (z, y, x) values.
track = [tr for tr in track if len(tr)>0]

blobs = [tr[0] for tr in track]

bboxes = []
for ts in track:
	bboxes.append(bounding_box(ts[0]))


centroids = [centroid(b) for b in blobs]

new_shape = (10, 100, 100) # z, y, x 

time = len(track)
depth = 12
height = 100
width = 100
newshape = (time, depth, 1, height, width)

midy = 50 
midx = 50 

centered_blobs = adjust_track(blobs, (5, 50, 50))

canvas = np.zeros(newshape, dtype='uint8')

for t in range(time):
	pix_list = centered_blobs[t]
	for z, y, x in pix_list:
		canvas[t, z, 0, y, x] = 255

outfile = 'sort5_cell_0.tif'
tf.imsave(outfile, canvas, imagej=True)







