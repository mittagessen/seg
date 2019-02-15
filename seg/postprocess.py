import math
import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter

from skimage.draw import line
from skimage.graph import MCP_Connect
from skimage.measure import approximate_polygon
from skimage.transform import estimate_transform
from skimage.morphology import skeletonize_3d

from sklearn.metrics.pairwise import euclidean_distances

from itertools import combinations
from collections import defaultdict

def denoising_hysteresis_thresh(im, low, high, sigma):
    im = gaussian_filter(im, sigma)
    lower = im > low
    components, count = label(lower, np.ones((3, 3)))
    valid = np.unique(components[lower & (im > high)])
    lm = np.zeros((count + 1,), bool)
    lm[valid] = True
    return lm[components]


def vectorize_lines(im: np.ndarray):
    """
    Vectorizes lines from a binarized array.
    """

    line_skel = skeletonize_3d(im)
    # find extremities by convolving with 3x3 filter (value == 2 on the line because of
    # 8-connected skeleton)
    line_skel = line_skel > 0
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    line_extrema = np.transpose(np.where((convolve2d(line_skel, kernel, mode='same') == 11) * line_skel))

    # this is the ugly hack from dhSegment. Instead calculating the graph
    # diameter to find the centerline of the skeleton (which is unbearably
    # slow) just take the two points with the largest euclidian distance as
    # endpoints. This breaks down in case of folded or spiral lines as the true
    # end points are closer closer than random branches on the skeleton.
    candidates = defaultdict(list)
    label_im, _ = label(line_skel, structure=np.ones((3, 3)))
    for pt in line_extrema:
        candidates[label_im[tuple(pt)]].append(pt)
    cc_extrema = []
    for pts in candidates.values():
        distance = euclidean_distances(np.stack(pts), np.stack(pts))
        i, j = np.unravel_index(distance.argmax(), distance.shape)
        cc_extrema.append(pts[i])
        cc_extrema.append(pts[j])

    class LineMCP(MCP_Connect):
        def __init__(self, *args, **kwargs):
           super().__init__(*args, **kwargs)
           self.connections = dict()
           self.scores = defaultdict(lambda: np.inf)

        def create_connection(self, id1, id2, pos1, pos2, cost1, cost2):
            k = (min(id1, id2), max(id1, id2))
            s = cost1 + cost2
            if self.scores[k] > s:
                self.connections[k] = (pos1, pos2, s)
                self.scores[k] = s

        def get_connections(self):
            results = []
            for k, (pos1, pos2, s) in self.connections.items():
                results.append(np.concatenate([self.traceback(pos1), self.traceback(pos2)[::-1]]))
            return results

        def goal_reached(self, int_index, float_cumcost):
            return 2 if float_cumcost else 0

    mcp = LineMCP(~line_skel)
    mcp.find_costs(cc_extrema)
    # incredibly slow graph diameter extraction
    #le = 0
    #idx = 0
    #for line in cc_extrema.values():
    #    le += len(list(combinations(line, 2)))
    #for line in cc_extrema.values():
    #    path = []
    #    mcp = LineMCP(~line_skel)
    #    for start, end in combinations(line, 2):
    #        mcp.find_costs([start, end])
    #        l = mcp.get_connections()[0]
    #        if len(l) > len(path):
    #            path = l
    #        print('{}/{}'.format(idx, le))
    #        idx += 1
    #    connections.append(path)
    # subsample lines using Douglas-Peucker
    return [approximate_polygon(line, 3).tolist() for line in mcp.get_connections()]


def line_extractor(im: np.ndarray, polyline: np.ndarray, context: int):
    """
    Args:
        im (np.ndarray):
        polyline (np.ndarray): Array of point (n, 2) defining a polyline.
        context (int): Padding around baseline for extraction.

    Returns:
        A dewarped image of the line.
    """
    # acquire control points by sampling the polyline, find perpendicular
    # points at distance `context` for each sample and estimate a
    # piecewise-affine transformation from those.
    def _unit_ortho_vec(p1, p2):
        vy = p1[0] - p2[0]
        vx = p1[1] - p2[1]
        dist = math.sqrt(vx**2 + vy**2)
        return (vx/dist, vy/dist)

    # pick start of line as left end for now (folded lines are problematic)
    if polyline[0][1] > polyline[-1][1]:
        polyline = list(reversed(polyline))
    upper_pts = []
    lower_pts = []
    for lineseg in zip(polyline, polyline[1::]):
        # calculate orthogonal vector
        uy, ux = _unit_ortho_vec(*lineseg)
        samples = line(lineseg[0][0], lineseg[0][1], lineseg[1][0], lineseg[1][1])
        lower = samples[0] - int(context * uy), samples[1] + int(context * ux)
        upper = samples[0] + int(context * uy), samples[1] - int(context * ux)
        for l in zip(*upper, *samples):
            upper_pts.append(np.transpose(line(*l))[-context//2:, :].T)
        for l in zip(*samples, *lower):
            lower_pts.append(np.transpose(line(*l))[1:context//2+1, :].T)
    return np.stack(im[x.tolist()] for x in np.concatenate((upper_pts, lower_pts), axis=2)).T
