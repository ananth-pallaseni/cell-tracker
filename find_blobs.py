# Locates blobs in a 4d stack
# Uses connectedComponents and merges blobs across Z 
# Writes a json file containing the blobs

import tifffile as tf
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2

import json
import sys

def loadTiff(fname):
    imraw = tf.TiffFile(fname)
    im = imraw.asarray()
    imraw.close()
    return im

def mark_blobs_brute_force(stack3d):
	depth, width, height = stack3d.shape

	total_cc = np.zeros(stack3d.shape, dtype='uint32')
	for z in range(depth):
		num, cc = cv2.connectedComponents(stack3d[z], 8, cv2.CV_32S)
		total_cc[z] = cc 

	for z in range(1, depth):
		prev = total_cc[z-1]
		cur = total_cc[z]

		l_and = np.logical_and(prev, cur)
		merges = set(zip( cur[l_and], prev[l_and] ))
		offset = prev.max() + 1 #- len(merges)
		cur[cur > 0] += offset
		for fr, to in merges:
			total_cc[z][cur==fr+offset] = to

	total_num = total_cc.max()
	return total_cc, total_num

def enum_blobs(cc, num):
	blobs = []
	for idx in range(1, num+1):
	 	zs, xs, ys = np.where(cc == idx)
	 	coords = list(zip(zs.astype('uint32'), xs.astype('uint32'), ys.astype('uint32')))
	 	coords = [(int(q), int(w), int(e)) for (q,w,e) in coords]
	 	#print('coords type = ' + str(type(coords[0][0])))
	 	if len(coords) > 0:
	 		blobs.append(coords)
	return blobs

def find_blobs_3d(stack3d):
	cc, n = mark_blobs_brute_force(stack3d)
	return enum_blobs(cc, n)


# Locates all blobs in a threshed image
# Returns an array containing blob arrays, one for each time step
def blobs3d(stack4d, store_all=False):
	blob_stack = []

	t = 0
	for stack3d in stack4d:
		print('\rstarting t = ' + str(t), end='')
		blobs = find_blobs_3d(stack3d)
		if store_all:
			with open('blobs_at_time_' + str(t) + '.json', 'w') as f:
				json.dump(blobs, f)
		blob_stack.append(blobs)
		t += 1

	print('\rFinished blob search          ')

	return blob_stack

def main(infile, store_all=False):
	stack4d = loadTiff(infile)
	return blobs3d(stack4d, store_all=store_all)


if __name__ == '__main__':
	infile = None
	outfile = None
	store_all = False

	argc = len(sys.argv)
	if argc > 1:
		infile = sys.argv[1]
	if argc > 2:
		outfile = sys.argv[2]
	if argc > 3:
		if sys.argv[3] == 'store-all':
			store_all = True

	if infile is not None and outfile is not None:
		print('Loading File: ' + infile)
		blob_stack = main(infile, store_all=store_all)

		print('Writing to file: ' + outfile)
		with open(outfile, 'w') as f:
			json.dump(blob_stack, f)


	


# # Test code:
# a = np.zeros( (3, 10, 10) ,  dtype='uint8')
# a[0, 2:4, 2:4] = 255
# a[1, 1:3, 1:3] = 255 
# a[0, 7:9, 7:9] = 255
# a[1, 5:6, 5:6] = 255
# a[2, 2:4, 2:4] = 255
# a[2, 7:9, 7:9] = 255

# cc, n = mark_blobs_brute_force(a)
# bbs = enum_blobs(cc, n)






