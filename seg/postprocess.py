import numpy as np
from scipy.signal import convolve2d
from scipy.ndimage import label
from scipy.ndimage.filters import gaussian_filter

from skimage.draw import line
from skimage.graph import MCP_Connect
from skimage.measure import approximate_polygon
from skimage.transform import estimate_transform
from skimage.morphology import skeletonize

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
    Vectorizes lines from a binarized array. Inspired by the dhSegment package.
    """
    line_skel = skeletonize(im)
    # find extremities by convolving with 3x3 filter (value == 2 on the line because of
    # 8-connected skeleton)
    kernel = np.array([[1,1,1],[1,10,1],[1,1,1]])
    line_extrema = np.transpose(np.where((convolve2d(line_skel, kernel, mode='same') == 11) * line_skel))
    # find least cost path between extrema
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
    mcp.find_costs(line_extrema)
    # subsample lines using Douglas-Peucker
    return [approximate_polygon(line, 3).tolist() for line in mcp.get_connections()]


def line_extractor(im: np.ndarray, polyline: np.ndarray, context: int, ctl_point_sample=5):
    """
    Args:
        im (np.ndarray):
        polyline (np.ndarray): Array of point (n, 2) defining a polyline.
        context (int): Padding around baseline for extraction.
        ctl_point_sample (int): interval for control point sampling on the
                                baseline

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

    # start point the line is normalized on
    start = polyline[0]
    src_points = []
    upper_src_points = []
    lower_src_points = []
    dst_points = []
    for lineseg in zip(polyline, polyline[1::]):
        # calculate orthogonal vector
        uy, ux = _unit_ortho_vec(*lineseg)
        # sample at distance
        samples = np.transpose(line(lineseg[0][0], lineseg[0][1], lineseg[1][0], lineseg[1][1]))[::ctl_point_sample]
        src_points.extend(samples.tolist())
        # evaluate at sample interval
        lower = samples.T[0] - context/2 * uy, samples.T[1] + context/2 * ux
        lower_src_points.extend(np.stack(lower, axis=1).tolist())
        upper =  samples.T[0] + context/2 * uy, samples.T[1] - context/2 * ux
        upper_src_points.extend(np.stack(upper, axis=1).tolist())
    dst_points.extend(line(start[0], start[1], start[0], start[1] + ctl_point_sample * len(src_points))[::ctl_point_sample])
    # add control points beneath baseline
    src_points.extend(lower_src_points)
    dst_points.extend(line(start[0] + context/2, start[1], start[0] + context/2, start[1] + ctl_point_sample * len(src_points))[::ctl_point_sample])
    # add control points above baseline
    src_points.extend(upper_src_points)
    dst_points.extend(line(start[0] - context/2, start[1], start[0] - context/2, start[1] + ctl_point_sample * len(src_points))[::ctl_point_sample])
    transform = estimate_transform('piecewise-affine', src_points, dst_points)
