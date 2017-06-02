# Performs the algorithm in subback50-medblur5-timeavg-thresh25.py 
# on each slice of a full ome-tiff


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


# Subtract autofluoresence channel from cell channel
def sub_auto_channel(auto_ch, cell_ch):
    time, width, height = cell_ch.shape

    mask = auto_ch > cell_ch
    cell_ch[mask] = 0

    for t in range(time):
        antimask = auto_ch[t] <= cell_ch[t]
        cell_ch[t][antimask] = cell_ch[t][antimask] - auto_ch[t][antimask]

    return cell_ch



# Subtract the average intensity of pixels across all time from each depth slice
# Note that this modifies the input image
def sub_time_avg(im, time):
    # create time average matrix
    tavg = np.sum(im, axis=0)
    tavg = tavg / time 
    tavg = tavg.astype('uint8')
    
    # Set all areas where tavg > im to 0 
    mask = tavg > im
    im[mask] = 0
    
    # For all areas where tavg <= im, subtract the two
    # Done in a for loop to avoid horrific memory crashes
    for t in range(time):
        antimask = tavg <= im[t,:,:]
        im[t][antimask] = im[t][antimask] - tavg[antimask]
    return im 

def subback50(stack):
    kernel = np.ones( (50,50), dtype='uint8')
    numslices = stack.shape[0]
    for i in range(numslices):
        im = stack[i]
        nim = cv2.blur(im, (3,3))
        nim = cv2.morphologyEx(nim, cv2.MORPH_TOPHAT, kernel)
        stack[i] = nim 
    return stack

def medblur5(stack):
    numslices = stack.shape[0]
    for i in range(numslices):
        im = stack[i]
        nim = cv2.medianBlur(im, 5)
        stack[i] = nim 
    return stack

def thresh25(stack):
    for t in range(stack.shape[0]):
        ret, thresh = cv2.threshold(stack[t], 25, 255, cv2.THRESH_BINARY)
        stack[t] = thresh
    return stack 

def adaptive_thresh(stack, block=51):
    for t in range(stack.shape[0]):
        thresh = cv2.adaptiveThreshold(stack[t], 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, 0)
        stack[t] = thresh
    return stack

def erode3(stack):
    kernel = np.ones((3,3), dtype='uint8')
    for t in range(stack.shape[0]):
        im = stack[t]
        stack[t] = cv2.erode(im, kernel)
    return stack 


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

    return im 


# Peforms bleach correction on im
# im has shape (time, height, width)
def bleach_correct(im):
    time = im.shape[0]
    ref = im[0]
    for t in range(time):
        im[t] = histogram_match(im[t], ref)
    return im 


# Last working copy:
def process_stack(stack, auto_stack=None):
    if auto_stack is not None: 
        stack = sub_auto_channel(auto_stack, stack)
    stack = subback50(stack)
    stack = medblur5(stack)
    stack = sub_time_avg(stack, stack.shape[0])
    stack = thresh25(stack)
    # stack = erode3(stack)
    return stack 


# Experimental version:
# def process_stack(stack, auto_stack=None):
#     stack = sub_time_avg(stack, stack.shape[0])
#     # if auto_stack is not None: 
#     #     stack = sub_auto_channel(auto_stack, stack)
#     # stack = medblur5(stack)
#     # stack = adaptive_thresh(stack)
#     stack = subback50(stack)
#     stack = medblur5(stack)
#     # stack = sub_time_avg(stack, stack.shape[0])
#     # stack = thresh25(stack)
#     # stack = erode3(stack)
#     return stack 


def process(whole_stack, whole_auto_stack=None):
    time, depth, x, y = whole_stack.shape 
    for d in range(depth):
        stack = whole_stack[:,d,:,:]
        auto_stack = whole_auto_stack[:,d,:,:]
        stack = process_stack(stack, auto_stack=auto_stack)
        whole_stack[:,d,:,:] = stack 
    return whole_stack


def save(stack, outfile):
    tf.imsave(outfile, stack, planarconfig='planar')

def main(infile, auto_ch=None):
    instack = loadTiff(infile)
    auto_stack = None
    if auto_ch is not None:
        auto_stack = loadTiff(auto_ch)
    print('Preprocessing')
    outstack = process(instack, whole_auto_stack=auto_stack)
    return outstack


if __name__ == '__main__':
    argc = len(sys.argv)
    infile = None
    outfile = None
    auto_ch = None

    argc = len(sys.argv)
    if argc > 1:
        infile = sys.argv[1]
        outfile = infile[:-4] + '-thresh.tif'
    if argc > 2:
        outfile = sys.argv[2]
    if argc > 3:
        auto_ch = sys.argv[3]

    if infile is not None and outfile is not None:
        outstack = main(infile, auto_ch=auto_ch)
        save(outstack, outfile)
