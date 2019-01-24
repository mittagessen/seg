import numpy as np
from scipy.ndimage.filters import gaussian_filter
from scipy.ndimage import label
from scipy.signal import convolve2d

from skimage.morphology import skeletonize
from skimage.graph import MCP_Connect

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
