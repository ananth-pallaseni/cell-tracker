
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

# Get the volume in unit coordinates of a bounding box
def unit_volume(bbox):
    dz = max(bbox[1] - bbox[0], 1)
    dy = max(bbox[3] - bbox[2], 1)
    dx = max(bbox[5] - bbox[4], 1)
    vol = dz * dy * dx 
    return vol

def raw_volume(plist):
	zl = 5
	yl = 273.9877 / 495
	xl = 278.4158 / 503

	# Volume of a pixel
	voxel = zl * yl * xl 

	# Volume of blob = num pixels * piel vol
	rvol = voxel * len(plist)


# Get the volume in microns of a bounding box
def mic_volume(bbox):
    zl = 5
    yl = 273.9877 / 495
    xl = 278.4158 / 503
    dz = (bbox[1] - bbox[0]) * zl
    if dz == 0:
        dz = 1
    dy = (bbox[3] - bbox[2]) * yl
    if dy == 0:
        dy = 1
    dx = (bbox[5] - bbox[4]) * xl
    if dx == 0:
        dx = 1
    vol = dz * dy * dx
    return vol

def bbox_stats(bbox):
	zl = 5
	yl = 273.9877 / 495
	xl = 278.4158 / 503
	s = {}
	s['dz_unit'] = max(bbox[1] - bbox[0] + 1, 1) # Add 1 to account for the following: if top z=3 and bot z=1 it exists across 3 z-slices but 3-1=2
	s['dy_unit'] = max(bbox[3] - bbox[2] + 1, 1) # Add 1 to account for the following: if top z=3 and bot z=1 it exists across 3 z-slices but 3-1=2
	s['dx_unit'] = max(bbox[5] - bbox[4] + 1, 1) # Add 1 to account for the following: if top z=3 and bot z=1 it exists across 3 z-slices but 3-1=2
	s['dz_mic'] = s['dz_unit'] * zl
	s['dy_mic'] = s['dy_unit'] * yl
	s['dx_mic'] = s['dx_unit'] * xl
	s['unit_volume'] = unit_volume(bbox)
	s['mic_volume'] = mic_volume(bbox)
	s['unit_area'] = s['dy_unit'] * s['dx_unit']

	return s

def blob_stats(blob, zl=5, yl=0.5535, xl=0.5535):
	s = {}

	# Raw blob stats
	s['raw_volume'] = raw_volume(blob)
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
	s['centroid'] = (sumz, sumy, sumx)
	

	# Get bounding box 
	zs = [b[0] for b in blob]
	minz = min(zs)
	maxz = max(zs)
	ys = [b[1] for b in blob]
	miny = min(ys)
	maxy = max(ys)
	xs = [b[2] for b in blob]
	minx = min(xs)
	maxx = max(xs)
	bbox = (minz, maxz, miny, maxy, minx, maxx)

	# bounding box stats
	s['bbox'] = bbox
	s['dz_unit'] = max(bbox[1] - bbox[0] + 1, 1) # Add 1 to account for the following: if top z=3 and bot z=1 it exists across 3 z-slices but 3-1=2
	s['dy_unit'] = max(bbox[3] - bbox[2] + 1, 1) # Add 1 to account for the following: if top z=3 and bot z=1 it exists across 3 z-slices but 3-1=2
	s['dx_unit'] = max(bbox[5] - bbox[4] + 1, 1) # Add 1 to account for the following: if top z=3 and bot z=1 it exists across 3 z-slices but 3-1=2
	s['dz_mic'] = s['dz_unit'] * zl
	s['dy_mic'] = s['dy_unit'] * yl
	s['dx_mic'] = s['dx_unit'] * xl
	s['unit_volume'] = unit_volume(bbox)
	s['mic_volume'] = mic_volume(bbox)
	s['unit_area'] = s['dy_unit'] * s['dx_unit']
	s['bbox_centroid'] = (s['dz_unit']/2, s['dy_unit']/2, s['dx_unit']/2)

	return s 



def filter_bbox(bb_stats, min_unit_area=100, max_unit_area=3000, min_unit_stack=1, max_unit_stack=10, ap_ratio=3, min_vol=200, max_vol=8000):

	min_area = bb_stats['unit_area'] > min_unit_area
	max_area = bb_stats['unit_area'] < max_unit_area
	area = min_area and max_area

	min_depth = bb_stats['dz_unit'] > min_unit_stack
	max_depth = bb_stats['dz_unit'] < max_unit_stack
	depth = min_depth and max_depth

	# volume = bb_stats['raw_volume'] < max_vol and bb_stats['raw_volume'] > min_vol

	inv_ap = 1 / ap_ratio
	aspect_ratio_xy = bb_stats['dx_unit'] / bb_stats['dy_unit'] < ap_ratio and bb_stats['dx_unit'] / bb_stats['dy_unit'] > inv_ap
	aspect_ratio = aspect_ratio_xy # and aspect_ratio_xz and aspect_ratio_yz

	return area and depth and aspect_ratio #and volume





def filter_blobs(all_blobs, zl=5, yl=0.5535, xl=0.5535):
	filtered = []
	for t in range(len(all_blobs)):
		# Grab blob dictionary for time t
		blob_slice = all_blobs[t]

		# # Get bounding boxes at time t
		# bboxes = get_bounding_boxes(blob_slice)

		# # Filter out blobs
		# sig = zip(blob_slice, bboxes)
		# sig = [b[0] for b in sig if filter_bbox(bbox_stats(b[1]))]

		sig = [b for b in blob_slice if filter_bbox(blob_stats(b, zl=zl, yl=yl, xl=xl))]

		# Add to overall list 
		filtered.append(sig)

	return filtered



if __name__ == '__main__':
	infile = None
	outfile = None

	argc = len(sys.argv)
	if argc > 1:
		infile = sys.argv[1]
	if argc > 2:
		outfile = sys.argv[2]

	if infile is not None and outfile is not None:
		all_blobs = []
		with open(infile) as f:
			all_blobs = json.load(f)
			
		filtered = filter_blobs(all_blobs)

		with open(outfile, 'w') as f:
			json.dump(filtered, f)

