import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg

import json
import math 
import sys

def bounding_ellipse(blob):
	pass

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


def plotEllipsoid(center, radii, rotation, ax=None, plotAxes=False, cageColor='b', cageAlpha=0.2):
    """Plot an ellipsoid"""
    make_ax = ax == None
    if make_ax:
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        
    u = np.linspace(0.0, 2.0 * np.pi, 100)
    v = np.linspace(0.0, np.pi, 100)
    
    # cartesian coordinates that correspond to the spherical angles:
    x = radii[0] * np.outer(np.cos(u), np.sin(v))
    y = radii[1] * np.outer(np.sin(u), np.sin(v))
    z = radii[2] * np.outer(np.ones_like(u), np.cos(v))
    # rotate accordingly
    for i in range(len(x)):
        for j in range(len(x)):
            [x[i,j],y[i,j],z[i,j]] = np.dot([x[i,j],y[i,j],z[i,j]], rotation) + center

    if plotAxes:
        # make some purdy axes
        axes = np.array([[radii[0],0.0,0.0],
                         [0.0,radii[1],0.0],
                         [0.0,0.0,radii[2]]])
        # rotate accordingly
        for i in range(len(axes)):
            axes[i] = np.dot(axes[i], rotation)


        # plot axes
        for p in axes:
            X3 = np.linspace(-p[0], p[0], 100) + center[0]
            Y3 = np.linspace(-p[1], p[1], 100) + center[1]
            Z3 = np.linspace(-p[2], p[2], 100) + center[2]
            ax.plot(X3, Y3, Z3, color=cageColor)

    # plot ellipsoid
    ax.plot_wireframe(x, y, z,  rstride=4, cstride=4, color=cageColor, alpha=cageAlpha)
    
    if make_ax:
        plt.show()
        plt.close(fig)
        del fig


def plot_eb(blob, center, rad, rot):
	P = blob 
	fig = plt.figure()
	ax = fig.add_subplot(111, projection='3d')

	# plot points
	ax.scatter(P[:,0], P[:,1], P[:,2], color='g', marker='*', s=100)

	# plot ellipsoid
	plotEllipsoid(center, rad, rot, ax=ax, plotAxes=True)
	    
	plt.show()
	plt.close(fig)
	del fig


def calc_ap(radii):
	_ = sorted(radii, reverse=True)[:2]
	return _[0] / _[1]

def get_ap_ratios(track):
	blobs = [np.array(ts[0]) for ts in track]
	radii = [getMinVolEllipse(b)[1] for b in blobs]
	aps = [calc_ap(r) for r in radii]
	return aps 


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

# Very naive - uses the distance travelled in the last timestep / timestep. First speed is 0
def get_speeds(track, timestep):
	blobs = [ts[0] for ts in track]
	cents = np.array([centroid(b) for b in blobs])
	diffs = cents[1:] - cents[:-1]
	dists = linalg.norm(diffs, ord=2, axis=1)
	speeds = dists / timestep
	speeds = np.concatenate([np.array([0]), speeds])
	return speeds
	

def process_track(track):
	track = [tt for tt in track if len(tt) > 0]
	aps = get_ap_ratios(track)
	spd = get_speeds(track, 3)
	return spd, aps 

if __name__ == '__main__':
	if len(sys.argv) > 1 and sys.argv[1] == 'all':
		all_blobs = []
		with open('all_blobs.json') as f:
			all_blobs = json.load(f)
		b = np.array(all_blobs[0][0])
		center, rad, rot = getMinVolEllipse(b)
		plot_eb(b, center, rad, rot)

		with open('individual_track0.json') as f:
			t0 = json.load(f)
		# t0 = [tt for tt in t0 if len(tt) > 0]
		# aps = get_ap_ratios(t0)
		# spd = get_speeds(t0, 3)

		spd, aps = process_track(t0)
		plt.plot(spd, aps, 'ro')
		plt.show()

	if len(sys.argv) > 1 and sys.argv[1] == 'temp':
		fblobs = []
		with open('filtered_blobs.json') as f:
			fblobs = json.load(f)

		tracks = []
		nm = 'temp_track_'
		for ii in range(10):
			fname = nm + str(ii) + '.json'
			with open(fname) as f:
				tracks.append(json.load(f))

		btracks = [[[fblobs[lst[0][0]][lst[0][1]]] for lst in tr] for tr in tracks]

		data = [process_track(tr) for tr in btracks]

		ii = 0
		for spd, asp in data:
			plt.clf()
			plt.xlabel('Speed (microns/s)')
			plt.ylabel('Aspect Ratio')
			plt.axis([0, 10, 0, 3])
			plt.plot(spd, np.array(asp), 'ro')
			plt.savefig('asp_' + str(ii) + '.png', bbox='tight')
			ii += 1




