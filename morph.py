



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

def surface_area(blob):
	bbox = get_bbox(blob)
	dims = bbox[1]-bbox[0], bbox[3] - bbox[2], bbox[5] - bbox[4]
	box = np.zeros(dims)
	box[blob] = 1

	