

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
	sumz = sumz / len(blob)
	sumy = sumy / len(blob) 
	sumx = sumx / len(blob)
	return (sumz, sumy, sumx)

# Finds euclidean distance between centroids
def centroid_diff(c1, c2):
	dz = c1[0] - c2[0]
	dy = c1[1] - c2[1]
	dx = c1[2] - c2[2]
	return math.sqrt(dz**2 + dy**2 + dx**2)


def find_top_matches(cur_cont, next_list):
	diffs = [(tup[0], centroid_diff(cur_cont, tup[2])) for tup in next_list]
	diffs = sorted(diffs, key=lambda x:x[1])
	return diffs


# Connect blobs through time 
def connect_blobs(all_blobs, max_dist=30, keep_top=None):
	time = len(all_blobs)

	# Initiate blob details. Detail format = (blob, time, cnetroid, [(index_in_next, best_diff)], prev_exists, blob_vol)
	blob_details = []
	for t in range(time):
		time_stack = []
		for bidx in range(len(all_blobs[t])):
			time_stack.append( [bidx, t, centroid(all_blobs[t][bidx]), [], False, len(all_blobs[t][bidx])] )
		blob_details.append(time_stack)

	# Go through each time_slice
	for i in range(time-1):
		t_cur = time - i - 2
		t_next = time - i - 1

		# For each blob_detail in current time slice
		for cur_blob_id in range(len(blob_details[t_cur])):
			tup = blob_details[t_cur][cur_blob_id]
			b = tup[0]
			t = tup[1]
			cnt = tup[2]
			idx_list = tup[3]
			prev = tup[4]

			# Find the closest blobs in next time slice
			cent_diffs = find_top_matches(cnt, blob_details[t_next])

			# Filter out matches that are too far
			cent_diffs = [tup for tup in cent_diffs if tup[1] < max_dist]

			# Keep top blobs
			if keep_top is not None:
				cent_diffs = cent_diffs[:keep_top]

		
			blob_details[t_cur][cur_blob_id][3] = cent_diffs
			for (nid, _) in cent_diffs:
				blob_details[t_next][nid][4] = True 

	return blob_details

def filter_connected(blob_details):
	filt_blob_details = []

	for t_slice in range(len(blob_details)-1):
		cur_slice = blob_details[t_slice]

		# Create a list such that rev[i] = all blobs pointing to index i 
		rev = [[] for i in blob_details[t_slice+1]]
		for bd in cur_slice:
			diff_list = bd[3]
			if len(diff_list) > 0:
				for (nid, diff) in diff_list:
					if nid >= 0: # phantom check: should never fire, but kept due to paranoia
						rev[nid].append((bd, diff))
					

		# Reduce rev so that each entry has at most one blob (the closest)
		for i in range(len(rev)):
			l = rev[i]
			if len(l) > 0:
				closest = min(l, key=lambda x: x[1])[0]
				rev[i] = closest

		# Create new blob_details list reflecting the relatioships in rev (each blob can only have one blob pointing to it)
		new_cur_slice = []
		for bd in cur_slice: # Go through each blob
			diff_list = bd[3]
			if len(diff_list) == 0: # Is terminus, append blob as is
				new_cur_slice.append(bd.copy())
			else: # Has pointers to next blobs. Might need to filter these
				# Filter diff_list, removing any nid that does not recognize bd as being its closest
				diff_list = [(nid, diff) for nid,diff in diff_list if rev[nid][0] == bd[0]]
				nbd = bd.copy()
				nbd[3] = diff_list
				new_cur_slice.append(nbd)

		# Make sure that list is ordered such that new_cur_slice[i][0] = i
		new_cur_slice = sorted(new_cur_slice, key=lambda x: x[0]) # Phantom operation: should not be needed, but kept due to paranoia

		filt_blob_details.append(new_cur_slice)

	# Append final slice manually
	filt_blob_details.append(blob_details[-1])

	return filt_blob_details
				

def create_track_helper(blob_details, t, idx):
	# Initialize track. Has the format [ [(time, id)] ] so that track[t][i] is the ith blob at time t
	track = []

	# Initialise queue
	q = [(t, idx)]

	# While queue is not empty
	while len(q) > 0:
		# Pop latest item
		cur_t, cur_id = q.pop()

		# Check if this time step exists in the track
		if len(track) <= cur_t:
			# If not then increase the size of track and add this item
			track.append( [(cur_t, cur_id)] )
		else:
			# If so, then add this item to that time step
			track[cur_t].append( (cur_t, cur_id) )

		# Add all the next blobs to the queue
		for tup in blob_details[cur_t][cur_id][3]:
			q.append( (cur_t + 1, tup[0]) )


	# For each time step take at most 2 of the largest blobs 
	track = [sorted(tt, key=lambda x: blob_details[x[0]][x[1]][5], reverse=True)[:2] for tt in track]

	return track


# Convert from a list of blob details to a list of tracks
def create_tracks(blob_details):
	all_tracks = []

	# Go through all time slices
	for t_slice in range(len(blob_details)):
		# Go through each blob
		for tup in blob_details[t_slice]:
			# If this blob has no blobs that point to it, then start a track
			if not tup[4]:
				# A track has the form such that track[i] = list of (time, id) tuples representing all blobs in the ith timestep of this track
				track = create_track_helper(blob_details, tup[1], tup[0])
				all_tracks.append(track)


	return all_tracks


def tracks_to_blobs(all_blobs, tracks):
	# Flatten out tracklist into list of (time, id) tuples 
	tempp = [tup for tr in tracks for tuplist in tr for tup in tuplist]

	# Create new blob list containg every pixel in every blob in tempp
	fblobs = [[all_blobs[bt][bid] for (bt, bid) in tempp if bt == t] for t in range(len(all_blobs))]

	return fblobs


# Naive filter - only removes blobs tht have not moved a certain distance across their whole lifetime
# ignores splits in the track -> takes the first blob in each time step
def filter_track(all_blobs, track, disp_thresh=10):
	# track has form such that track[i] is a list containing (t, id) tuples for time step i of the track

	# Track displacement over all time steps
	lifetime_disp = 0

	# Grab first centroid in track
	bt, bid = track[0][0]
	last_pos = centroid(all_blobs[bt][bid])

	# Go through all time steps in track
	for time_step in track:
		# Get the time and id of the first blob in this time step
		bt, bid = time_step[0]

		# Grab the blob out of the all_blobs array
		blob = all_blobs[bt][bid]

		# Compute centroid
		cent = centroid(blob)

		# Find distance from last pos
		disp = centroid_diff(cent, last_pos)

		lifetime_disp += disp 
		last_pos = cent 
		

	if lifetime_disp > disp_thresh:
		return True
	else:
		return False


def main(all_blobs, keep_top=2, len_thresh=10, disp_thresh=100, seperate_tracks=False):

	# Create blob connection details
	bd = connect_blobs(all_blobs, keep_top=keep_top)

	# Process connection details so that each blob has at most one blob pointing at it
	fblobs = filter_connected(bd)

	# Convert connection details into tracks
	tracks = create_tracks(fblobs)

	# Filter out tracks smaller than len_thresh
	ftracks = [tr for tr in tracks if len(tr) >= len_thresh]

	# Filter out tracks by lifetime displacement
	ftracks = [tr for tr in tracks if filter_track(all_blobs, tr, disp_thresh=disp_thresh)]

	print('Found ' + str(len(ftracks)) + ' tracks')

	# Filter tracks by size and convert back into pixels
	tblobs = tracks_to_blobs(all_blobs, ftracks)

	# Seperate into individual tracks
	if seperate_tracks:
		for i in range(len(ftracks)):
			tr = ftracks[i]
			sep_blobs = tracks_to_blobs(all_blobs, [tr])

			sep_name = 'individual_track' + str(i) + '.json'
			with open(sep_name, 'w') as f:
				json.dump(sep_blobs, f)

	return tblobs


# #########################################

# track0 = filt_tracks[0] 
# track0blobs = [tup for lst in track0 for tup in lst]
# track0blobs = sorted(track0blobs, key=lambda x:x[1])
# track0blobs = [[all_blobs[bt][bid] for (bt, bid) in track0blobs if bt == t] for t in range(len(all_blobs))]
# with open('track0.json', 'w') as f:
# 	json.dump(track0blobs, f)

# track1 = filt_tracks[1] 
# track1blobs = [tup for lst in track1 for tup in lst]
# track1blobs = sorted(track1blobs, key=lambda x:x[1])
# track1blobs = [[all_blobs[bt][bid] for (bt, bid) in track1blobs if bt == t] for t in range(len(all_blobs))]
# with open('track1.json', 'w') as f:
# 	json.dump(track1blobs, f)

# #########################################

if __name__ == '__main__':
	infile = None
	outfile = None
	len_thresh = 10
	seperate_tracks = False

	argc = len(sys.argv)
	if argc > 1:
		infile = sys.argv[1]
	if argc > 2:
		outfile = sys.argv[2]
	if argc > 3:
		len_thresh = int(sys.argv[3])
	if argc > 4 and sys.argv[4] == 'seperate':
		seperate_tracks = True

	if infile is not None and outfile is not None:
		# Load input blob file
		all_blobs = []
		print('Loading ' + infile)
		with open(infile) as f:
			all_blobs = json.load(f)

		tblobs = main(all_blobs, len_thresh=len_thresh, seperate_tracks=seperate_tracks)

		# Save 
		print('Saving to ' + outfile)
		with open(outfile, 'w') as f:	
			json.dump(tblobs, f)