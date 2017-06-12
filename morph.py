import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg

import json
import math 
import sys



def getMinVolEllipse(P, tolerance=0.01):
    """ Find the minimum volume ellipsoid which holds all the points
    
    Based on work by Nima Moshtagh
    http://www.mathworks.com/matlabcentral/fileexchange/9542
    and also by looking at:
    http://cctbx.sourceforge.net/current/python/scitbx.math.minimum_covering_ellipsoid.html
    Which is based on the first reference anyway!
    
    Here, P is a numpy array of N dimensional points like this:
    P = [[x,y,z,...], <-- one point per line
         [x,y,z,...],
         [x,y,z,...]]
    
    Returns:
    (center, radii, rotation)
    
    """
    (N, d) = np.shape(P)
    d = float(d)

    # Q will be our working array
    Q = np.vstack([np.copy(P.T), np.ones(N)]) 
    QT = Q.T
    
    # initializations
    err = 1.0 + tolerance
    u = (1.0 / N) * np.ones(N)

    # Khachiyan Algorithm
    while err > tolerance:
        V = np.dot(Q, np.dot(np.diag(u), QT))
        M = np.diag(np.dot(QT , np.dot(linalg.inv(V), Q)))    # M the diagonal vector of an NxN matrix
        j = np.argmax(M)
        maximum = M[j]
        step_size = (maximum - d - 1.0) / ((d + 1.0) * (maximum - 1.0))
        new_u = (1.0 - step_size) * u
        new_u[j] += step_size
        err = np.linalg.norm(new_u - u)
        u = new_u

    # center of the ellipse 
    center = np.dot(P.T, u)

    # the A matrix for the ellipse
    A = linalg.inv(
                   np.dot(P.T, np.dot(np.diag(u), P)) - 
                   np.array([[a * b for b in center] for a in center])
                   ) / d
                   
    # Get the values we'd like to return
    U, s, rotation = linalg.svd(A)
    radii = 1.0/np.sqrt(s)
    
    return (center, radii, rotation)


def get_bbox(blob):
	zs = [b[0] for b in blob]
	minz = min(zs)
	maxz = max(zs)

	ys = [b[1] for b in blob]
	miny = min(ys)
	maxy = max(ys)

	xs = [b[2] for b in blob]
	minx = min(xs)
	maxx = max(xs)

	return (minz, maxz, miny, maxy, minx, maxx)

# def surface_area(blob, zmic, ymic, xmic):
# 	bbox = get_bbox(blob)
# 	dims = bbox[1]-bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]
# 	box = np.zeros(dims)

# 	# Populate box with blob points
# 	blobz = [b[0] for b in blob]
# 	bloby = [b[1] for b in blob]
# 	blobx = [b[2] for b in blob]
# 	box[blobz, bloby, blobx] = 1







