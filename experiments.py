import tifffile as tf
# %matplotlib inline
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import cv2
import sys


# Extract the raw image data from an ome-tiff file
# Note: returns a numpy array in the format (time, z-val, y-val, x-val) with dtype=uint8
def loadTiff(fname):
    imraw = tf.TiffFile(fname)
    im = imraw.asarray()
    imraw.close()
    return im


def abs_diffs(stack4d):
	time = stack4d.shape[0]
	for t in range(time-1):
		stack4d[t] = cv2.absdiff(stack4d[t], stack4d[t+1])
	return stack4d


def abs_diff_step(stack4d, step=5):
	time = stack4d.shape[0]
	for t in range(time-step):
		stack4d[t] = cv2.absdiff(stack4d[t], stack4d[t+step])
	return stack4d


# def sub_time_average(stack4d, window=5):
# 	time = stack4d.shape[0]
# 	for t in range(window):
# 		tavg = np.sum(stack4d[t:t+window], axis=0) / window
# 		mask = tavg >= stack4d[t]
# 		stack4d[t][mask] = 0

def create_background(stack4d, step=5):
	time = stack4d.shape[0]
	for t in range(step, time-step):
		tavg = np.sum(stack4d[t-step:t+step], axis=0) / (2*step)
		stack4d[t] = tavg 

	for t in range(step):
		tavg = np.sum(stack4d[:t+step], axis=0) / (t+step)
		stack4d[t] = tavg 

	for t in range(time-step, time):
		tavg = np.sum(stack4d[t-step:], axis=0) / (step + time - t)
		stack4d[t] = tavg

	return stack4d


# Adjusts im so that it's histogram more closely matches ref
def histogram_match(im, ref):
    hist, bins = np.histogram(im.flatten(), 256, [0, 256])
    rhist, rbins = np.histogram(ref.flatten(), 256, [0, 256])

    cumhist = hist.cumsum()
    refcumhist = rhist.cumsum()

    table_pairs = []
    # prev = 1
    # for i in range(len(cumhist)):
    #     xi = cumhist[i]
    #     while prev < len(cumhist):
    #         yi = refcumhist[prev]
    #         if yi > xi:
    #             table_pairs.append( (i, prev-1) )
    #             break
    #         else:
    #             prev += 1

    prev = 0
    for i in range(len(cumhist)):
        xi = cumhist[i]

        # Find the smallest ref value that has cdf >= xi
        for j in range(prev, len(cumhist)):
            yi = refcumhist[j]
            if yi >= xi:
                prev = j
                pair = (i, j)
                table_pairs.append(pair)
                break
        



    for (frm, to) in table_pairs:
        im[im==frm] = to 


# Peforms bleach correction on im
# im has shape (time, height, width)
def bleach_correct(im):
    time = im.shape[0]
    ref = im[0]
    for t in range(time):
        im[t] = histogram_match(im[t], ref)
    return im 







if __name__ == '__main__':
	infile = None 
	outfile = None

	argc = len(sys.argv)

	if argc > 1:
		infile = sys.argv[1]
	if argc > 2:
		outfile = sys.argv[2]

	if infile is not None and outfile is not None:
		stack4d = loadTiff(infile)

		# Abs diff
		# stack4d = abs_diff_step(stack4d)

		# Create background 
		# stack4d = create_background(stack4d, step=50)

		# Overall background
		# stack4d = np.sum(stack4d, axis=0) / stack4d.shape[0]

		# Perform bleach correction
		time, depth, x, y = stack4d.shape 
		for z in range(depth):
			print('Starting depth {}'.format(z), end='\r')
			sl = stack4d[:,z]	
			stack4d[:,z] = bleach_correct(sl)





		tf.imsave(outfile, stack4d, planarconfig='planar')