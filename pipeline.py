import preprocess as mod_pre
import find_blobs as mod_find
import filter_blobs as mod_filter
import track_blobs_general as mod_track
import draw_blobs as mod_draw

import sys
from multiprocessing import Pool 
import json

def pipeline(infile, outfile, autofile=None):
	print('Loading ' + infile)
	q = mod_pre.main(infile, auto_ch=autofile)

	pre_fname = infile[:-4] + '-thresh.tif'
	print('Threshed file is ' + pre_fname)
	mod_pre.save(q, pre_fname)

	print('Finding blobs', end='')
	if __name__ == '__main__':
		print(': Using 8 processes...')
		q = [sl for sl in q]
		pool = Pool(processes=8)
		q = pool.map(mod_find.find_blobs_3d, q)
	else:
		print()
		q = mod_find.blobs3d(q)

	all_blobs_fname = 'all_blobs.json'
	print('Storing all blobs in {}'.format(all_blobs_fname))
	with open(all_blobs_fname, 'w') as f:
		json.dump(q, f)

	print('Filtering blobs')
	q = mod_filter.filter_blobs(q)

	filt_blobs_fname = 'filtered_blobs.json'
	print('Storing filtered blobs in {}'.format(filt_blobs_fname))
	with open(filt_blobs_fname, 'w') as f:
		json.dump(q, f)

	filt_fname = infile[:-4] + '-filtered.tif'
	print('Filtered file is ' + filt_fname)
	mod_draw.draw_copy_of(pre_fname, q, outfile=filt_fname)

	print('Creating tracks')
	q = mod_track.main(q, len_thresh=20, seperate_tracks=True, keep_top=1)

	mod_draw.draw_bboxes_from_file(pre_fname, q, outfile=outfile)


if __name__ == '__main__':
	infile = None
	outfile = None
	autofile = None

	argc = len(sys.argv)
	if argc > 1:
		infile = sys.argv[1]
	if argc > 2:
		outfile = sys.argv[2]
	if argc > 3:
		autofile = sys.argv[3]

	if infile is not None and outfile is not None:
		pipeline(infile, outfile, autofile=autofile)
