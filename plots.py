import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
from numpy import linalg

import json
import math 
import sys

# Taken from "https://github.com/minillinim/ellipsoid/blob/master/ellipsoid.py"
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



# Taken from "http://nicky.vanforeest.com/misc/fitEllipse/fitEllipse.html"
def fitEllipse(x,y):
    x = x[:,np.newaxis]
    y = y[:,np.newaxis]
    D =  np.hstack((x*x, x*y, y*y, x, y, np.ones_like(x)))
    S = np.dot(D.T,D)
    C = np.zeros([6,6])
    C[0,2] = C[2,0] = 2; C[1,1] = -1
    E, V =  linalg.eig(np.dot(linalg.inv(S), C))
    n = np.argmax(np.abs(E))
    a = V[:,n]
    return a


def ellipse_center(a):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    num = b*b-a*c
    x0=(c*d-b*f)/num
    y0=(a*f-b*d)/num
    return np.array([x0,y0])


def ellipse_angle_of_rotation( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    return 0.5*np.arctan(2*b/(a-c))


def ellipse_axis_length( a ):
    b,c,d,f,g,a = a[1]/2, a[2], a[3]/2, a[4]/2, a[5], a[0]
    up = 2*(a*f*f+c*d*d+g*b*b-2*b*d*f-a*c*g)
    down1=(b*b-a*c)*( (c-a)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    down2=(b*b-a*c)*( (a-c)*np.sqrt(1+4*b*b/((a-c)*(a-c)))-(c+a))
    res1=np.sqrt(up/down1)
    res2=np.sqrt(up/down2)
    return np.array([res1, res2])


def ellipse_2d(points):
    y = points[:, 1]
    x = points[:, 2]
    A = fitEllipse(x, y)
    ang = ellipse_angle_of_rotation(A)
    axlen = ellipse_axis_length(A)
    elcenter = ellipse_center(A)
    return [elcenter, axlen, ang]

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


# Taken from "https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249"
def unit_vector(vector):
    """ Returns the unit vector of the vector.  """
    return vector / np.linalg.norm(vector)

def angle_between(v1, v2):
    """ Returns the angle in radians between vectors 'v1' and 'v2'::

            >>> angle_between((1, 0, 0), (0, 1, 0))
            1.5707963267948966
            >>> angle_between((1, 0, 0), (1, 0, 0))
            0.0
            >>> angle_between((1, 0, 0), (-1, 0, 0))
            3.141592653589793
    """
    v1_u = unit_vector(v1)
    v2_u = unit_vector(v2)
    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

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

    if len(sys.argv) > 1 and sys.argv[1] == 'details5':
        print('details5')
        for i in range(2):
            with open('sort5_track_' + str(i) + '_details.json') as f:
                dlst = json.load(f)
            cents = np.array([s['centroid'] for s in dlst])
            diffs = cents[1:] - cents[:-1]
            diffs = linalg.norm(diffs, ord=2, axis=1)
            spds = diffs / 3 

            aps12 = [s['ellipse_aspect_ratio_13'] for s in dlst]

            plt.clf()
            plt.xlabel('Speed (microns/s)')
            plt.ylabel('Aspect Ratio (longest/third_longest)')
            plt.title('Sort 5 pos 2 - Track ' + str(i))
            plt.axis([0, 10, 0, 15])
            plt.plot(spds, aps12[1:], 'ro')
            plt.savefig('asp13_' + str(i) + '.png', bbox='tight')

    if len(sys.argv) > 1 and sys.argv[1] == 'details6':
        print('details6')
        for i in range(5):
            with open('sort6pos2_track_' + str(i) + '_stitched_details.json') as f:
                dlst = json.load(f)
            cents = np.array([s['centroid'] for s in dlst])
            diffs = cents[1:] - cents[:-1]
            diffs = linalg.norm(diffs, ord=2, axis=1)
            spds = diffs / 3 

            aps12 = [s['ellipse_aspect_ratio_13'] for s in dlst]

            plt.clf()
            plt.xlabel('Speed (microns/s)')
            plt.ylabel('Aspect Ratio (longest/third_longest)')
            plt.title('Sort 6 pos 2 - Track ' + str(i))
            plt.axis([0, 10, 0, 15])
            plt.plot(spds, aps12[1:], 'ro')
            plt.savefig('sort6pos2_asp13_' + str(i) + '.png', bbox='tight')


    if len(sys.argv) > 1 and sys.argv[1] == 'stitch':
        print('stitch')
        tracks = []
        for i in range(12):
            if i == 6 or i == 8:
                continue
            with open('sort6pos2_track_' + str(i) + '_details.json') as f:
                tr = json.load(f)
                tracks.append(tr) 


        for t in range(len(tracks)):
            tr = tracks[t]
            print('{0:<3} {1:>5} {2:>5} {3:>40} {4:>40}'.format(t, str(tr[0]['time']), str(tr[-1]['time']), str(tr[0]['centroid'][1:]), str(tr[-1]['centroid'][1:])))


        s1 = tracks[0] + tracks[3] + tracks[5] + tracks[6] + tracks[9]
        s2 = tracks[1] 
        s3 = tracks[2] + tracks[8]
        s4 = tracks[4] 
        s5 = tracks[7]
        stitched = [s1, s2, s3, s4, s5]

        # for i in range(len(stitched)):
        #     s = stitched[i]
        #     nm = 'sort6pos2_track_' + str(i) + '_stitched_details.json'
        #     with open(nm, 'w') as f:
        #         json.dump(s, f)

    if len(sys.argv) > 1 and sys.argv[1] == 'ap/t':
        print('speed')
        tracks5 = []
        tracks6 = []


        for i in range(2):
            with open('sort5_track_' + str(i) + '_details.json') as f:
                track = json.load(f)
                tracks5.append(track)


        for i in range(5):
            nm = 'sort6pos2_track_' + str(i) + '_stitched_details.json'
            with open(nm) as f:
                track = json.load(f)
                tracks6.append(track)

        for i in range(len(tracks5)):
            track = tracks5[i]
            ap12 = [det['ellipse_aspect_ratio_12'] for det in track]
            ap13 = [det['ellipse_aspect_ratio_13'] for det in track]
            time = [det['time'] for det in track]

            plt.clf()
            plt.xlabel('Time (3 minutes)')
            plt.ylabel('Aspect Ratio (longest/second_longest)')
            plt.title('Sort 5 pos 2 - Track ' + str(i))
            # plt.axis([0, 10, 0, 15])
            plt.plot(time, ap12, 'ro')
            plt.savefig('sort5_asp12_vs_time_' + str(i) + '.png', bbox='tight')

            plt.clf()
            plt.xlabel('Time (3 minutes)')
            plt.ylabel('Aspect Ratio (longest/third_longest)')
            plt.title('Sort 5 pos 2 - Track ' + str(i))
            # plt.axis([0, 10, 0, 15])
            plt.plot(time, ap13, 'ro')
            plt.savefig('sort5_asp13_vs_time_' + str(i) + '.png', bbox='tight')

        for i in range(len(tracks6)):
            track = tracks6[i]
            ap12 = [det['ellipse_aspect_ratio_12'] for det in track]
            ap13 = [det['ellipse_aspect_ratio_13'] for det in track]
            time = [det['time'] for det in track]

            plt.clf()
            plt.xlabel('Time (3 minutes)')
            plt.ylabel('Aspect Ratio (longest/second_longest)')
            plt.title('Sort 6 pos 2 - Track ' + str(i))
            # plt.axis([0, 10, 0, 15])
            plt.plot(time, ap12, 'ro')
            plt.savefig('sort6pos2_asp12_vs_time_' + str(i) + '.png', bbox='tight')

            plt.clf()
            plt.xlabel('Time (3 minutes)')
            plt.ylabel('Aspect Ratio (longest/third_longest)')
            plt.title('Sort 6 pos 2 - Track ' + str(i))
            # plt.axis([0, 10, 0, 15])
            plt.plot(time, ap13, 'ro')
            plt.savefig('sort6pos2_asp13_vs_time_' + str(i) + '.png', bbox='tight')


    if len(sys.argv) > 1 and sys.argv[1] == 'plot_test':

        # p1 = np.random.random( (200, 3) ) * 100
        p1 = np.reshape([np.random.random()*100 for i in range(300)],(100,3))
        p1[:,0] *= 0.1
        p1[:,1] *= 0.5 

        c1, rad1, rot1 = getMinVolEllipse(p1)
        amax = np.argmax(rad1)
        vec1 = [0, 0, 0]
        vec1[amax] = rad1[amax]
        vec1 = np.dot(vec1, rot1)
        vec2 = [0, vec1[1], vec1[2]]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(p1[:,0], p1[:,1], p1[:,2], color='y', marker='*')
        ax.plot( [c1[0], c1[0]+vec1[0]] , [c1[1], c1[1]+vec1[1]] , [c1[2], c1[2]+vec1[2]] , color='r')
        ax.plot( [c1[0], c1[0]+vec2[0]] , [c1[1], c1[1]+vec2[1]] , [c1[2], c1[2]+vec2[2]] , color='g')
        plotEllipsoid(c1, rad1, rot1, ax=ax, plotAxes=True)
        plt.show()

    if len(sys.argv) > 1 and sys.argv[1] == 'polarity_test':
        with open('sort6pos2_track_4_stitched_details.json') as f:
            track = json.load(f)

        # index = 1
        for index in range(7):
            cent0 = np.array(track[index]['centroid'])
            cent1 = np.array(track[index+1]['centroid'])
            diff = cent1 - cent0
            dir_ang = np.arctan(diff[1] / diff[2])

            ecent = track[index]['ellipse_center']
            rad = track[index]['ellipse_radii']
            rot = track[index]['ellipse_rot']
            amax = np.argmax(rad)
            v = [0,0,0]
            v[amax] = 10 
            v = np.dot(v, rot)
            pol_ang = np.arctan(v[1] / v[2])
            v_rev = [0,0,0]
            v_rev[amax] = -10
            v_rev = np.dot(v_rev, rot)
            rev_pol_ang = np.arctan(v[1]/v[2])

            # Taken from "https://stackoverflow.com/questions/2827393/angles-between-two-n-dimensional-vectors-in-python/13849249#13849249"
            def unit_vector(vector):
                """ Returns the unit vector of the vector.  """
                return vector / np.linalg.norm(vector)

            def angle_between(v1, v2):
                """ Returns the angle in radians between vectors 'v1' and 'v2'::

                        >>> angle_between((1, 0, 0), (0, 1, 0))
                        1.5707963267948966
                        >>> angle_between((1, 0, 0), (1, 0, 0))
                        0.0
                        >>> angle_between((1, 0, 0), (-1, 0, 0))
                        3.141592653589793
                """
                v1_u = unit_vector(v1)
                v2_u = unit_vector(v2)
                return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

            print('polarity = ' + str(pol_ang), v, angle_between(v, diff))
            print('revr pol = ' + str(rev_pol_ang), v_rev, angle_between(v_rev, diff))
            print('direction = ' + str(dir_ang), diff)

            if angle_between(v, diff) > angle_between(v_rev, diff):
                v = v_rev
                pol_ang = rev_pol_ang

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            # ax.axis([-2, 10, 50, 250, 50, 200])
            ax.set_xlim3d(-2, 10)
            ax.set_ylim3d(150,220)
            ax.set_zlim3d(150,220)
            # ax.scatter(p1[:,0], p1[:,1], p1[:,2], color='y', marker='*')

            # Red is polarity
            ax.plot( [ecent[0], ecent[0]+v[0]] , [ecent[1], ecent[1]+v[1]] , [ecent[2], ecent[2]+v[2]] , color='r')

            # # Yellow is reverse polarity
            # ax.plot( [ecent[0], ecent[0]+v_rev[0]] , [ecent[1], ecent[1]+v_rev[1]] , [ecent[2], ecent[2]+v_rev[2]] , color='y')

            # Green is direction
            ax.plot( [ecent[0], ecent[0]+diff[0]] , [ecent[1], ecent[1]+diff[1]] , [ecent[2], ecent[2]+diff[2]] , color='g')

            plotEllipsoid(ecent, rad, rot, ax=ax, plotAxes=True)
            plt.show()
            plt.clf()



    if len(sys.argv) > 1 and sys.argv[1] == 'polarity':
        print('polarity')
        tracks5 = []
        tracks6 = []


        for i in range(2):
            with open('sort5_track_' + str(i) + '_details.json') as f:
                track = json.load(f)
                tracks5.append(track)


        for i in range(5):
            nm = 'sort6pos2_track_' + str(i) + '_stitched_details.json'
            with open(nm) as f:
                track = json.load(f)
                tracks6.append(track)


    
        def get_polarity_vec(rad, rot):
            v = [0,0,0]
            amax = np.argmax(rad)
            v[amax] = 1
            v = np.dot(v, rot)
            return v 

        def get_polarity_2d(rad, rot):
            v = get_polarity_vec(rad, rot)
            v2d = [0, v[1], v[2]]
            ang = np.arctan(v[1] / v[2])
            return ang 

        def get_direction_vecs(track):
            cents = np.array( [det['centroid'] for det in track] )
            diffs = cents[1:] - cents[0:-1]
            return diffs 

        def get_directions(track):
            diffs = get_direction_vecs(track)
            angs = [np.arctan(v[1] / v[2]) for v in diffs]
            return angs           

        def calc_between_angle(tup):
            polvec, dirvec = tup 
            rev_pol = -polvec 
            normal = angle_between(polvec, dirvec)
            reverse = angle_between(rev_pol, dirvec)
            ret = polvec, normal
            if reverse < normal:
                ret = rev_pol, reverse
            return ret


        total_dirs = []
        total_pols = []


        # for i in range(len(tracks5)):
        #     track = tracks5[i]
        #     pol = [get_polarity_2d(det['ellipse_radii'], det['ellipse_rot']) for det in track]
        #     dirs = get_directions(track)

        #     total_pols += pol[:-1] 
        #     total_dirs += dirs

        #     plt.clf()
        #     plt.xlabel('Direction of movement in xy plane (rad)')
        #     plt.ylabel('Cell Polarity in xy plane (rad)')
        #     plt.title('Sort 5 - Track ' + str(i))
        #     plt.axis([-2.15, 2.15, -2.15, 2.15])
        #     plt.plot(dirs, pol[:-1], 'ro')
        #     plt.savefig('sort5_dir_vs_polar_' + str(i) + '.png', bbox='tight')

        # for i in range(len(tracks6)):
        #     track = tracks6[i]
        #     pol = [get_polarity_2d(det['ellipse_radii'], det['ellipse_rot']) for det in track]
        #     dirs = get_directions(track)

        #     total_pols += pol[:-1] 
        #     total_dirs += dirs

        #     plt.clf()
        #     plt.xlabel('Direction of movement in xy plane (rad)')
        #     plt.ylabel('Cell Polarity in xy plane (rad)')
        #     plt.title('Sort 6 Pos 2 - Track ' + str(i))
        #     plt.axis([-2.15, 2.15, -2.15, 2.15])
        #     plt.plot(dirs, pol[:-1], 'ro')
        #     plt.savefig('sort6pos2_dir_vs_polar_' + str(i) + '.png', bbox='tight')

        # plt.clf()
        # plt.xlabel('Direction of movement in xy plane (rad)')
        # plt.ylabel('Cell Polarity in xy plane (rad)')
        # plt.title('Sort 5 & 6 Pos 2 - All Tracks ')
        # plt.axis([-2.15, 2.15, -2.15, 2.15])
        # plt.plot(total_dirs, total_pols, 'ro')
        # plt.savefig('dir_vs_polar_all.png', bbox='tight')        

        total_angles = []
        
        for i in range(len(tracks5)):
            track = tracks5[i]
            pol = [get_polarity_vec(det['ellipse_radii'], det['ellipse_rot']) for det in track]
            dirs = get_direction_vecs(track)

            pzip = list(zip(pol[:-1], dirs))
            ang_lst = [calc_between_angle(tup)[1] for tup in pzip]
            total_angles += ang_lst

            plt.clf()
            plt.xlabel('Cell number')
            plt.ylabel('Angle between direction of motion and polarity (rad)')
            plt.title('Sort 5 - Track ' + str(i))
            # plt.axis([-2.15, 2.15, -2.15, 2.15])
            plt.plot(list(range(len(ang_lst))), ang_lst, 'ro')
            plt.savefig('sort5_angle_between_dir_and_polar_' + str(i) + '.png', bbox='tight')


        for i in range(len(tracks6)):
            print('sort6', i)
            track = tracks6[i]
            pol = [get_polarity_vec(det['ellipse_radii'], det['ellipse_rot']) for det in track]
            dirs = get_direction_vecs(track)

            pzip = list(zip(pol[:-1], dirs))
            ang_lst = [calc_between_angle(tup)[1] for tup in pzip]
            
            total_angles += ang_lst

            plt.clf()
            plt.xlabel('Cell number')
            plt.ylabel('Angle between direction of motion and polarity (rad)')
            plt.title('Sort 6 Pos 2 - Track ' + str(i))
            # plt.axis([-2.15, 2.15, -2.15, 2.15])
            plt.plot(list(range(len(ang_lst))), ang_lst, 'ro')
            plt.savefig('sort6pos2_angle_between_dir_and_polar_' + str(i) + '.png', bbox='tight')

        plt.clf()
        plt.xlabel('Cell number')
        plt.ylabel('Angle between direction of motion and polarity (rad)')
        plt.title('Sort 5 & 6 Pos 2 - All Tracks ')
        # plt.axis([-2.15, 2.15, -2.15, 2.15])
        plt.plot(list(range(len(total_angles))), total_angles, 'ro')
        plt.savefig('angle_between_dir_and_polar_all.png', bbox='tight')
























