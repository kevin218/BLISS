from scipy import spatial
from sklearn.externals import joblib
from pylab import *;

ion()
from time import time

y, x = 0, 1


def extractData(file):
    group = joblib.load(file)
    xcenters = group[b'centers'][0, :, :, x].flatten()
    ycenters = group[b'centers'][0, :, :, y].flatten()
    fluxes = group[b'phots'][0, -1].flatten()
    points = zip(xcenters, ycenters)
    return list(points), fluxes


# Procedure: nearest
# Parameters: point — coordinates of a point (list (x,y))
#             point_list — a list of coordinates
#             neighbors — How many neighbors to look for.
# Returns: the indices of the nearest neighbors in point_list to point.

def nearest(point, neighbors, tree):
    neighbors = tree.query(point, k=neighbors)
    return neighbors[1]


# Procedure: removeOutliers
# Parameters: point_list — a list of lists of coordinates ((x, y))
#             flux_list — a list of fluxes corresponding to every point to point_list
#             x_sigma_cutoff — number of accepted standard deviations in X (default= 3)
#             y_sigma_cutoff — number of accepted standard deviations in Y (default= 3)
# Returns: list of all points below [cutoff] standard deviations and with 13 < x, y < 17

def removeOutliers(point_list, flux_list, x_sigma_cutoff=3, y_sigma_cutoff=3):
    avg, sigma = mean(point_list, axis=0), std(point_list, axis=0)
    xmax = (avg[0] + x_sigma_cutoff * sigma[0])
    ymax = (avg[1] + y_sigma_cutoff * sigma[1])
    xmin = (avg[0] - x_sigma_cutoff * sigma[0])
    ymin = (avg[1] - y_sigma_cutoff * sigma[1])
    points = []
    fluxes = []
    i = 0
    for point in point_list:
        if point[0] < xmax and point[1] < ymax and point[0] > xmin and point[1] > ymin and point[0] < 17 \
                and point[1] < 17 and point[0] > 13 and point[1] > 13:
            points.append(point)
            fluxes.append(flux_list[i])
        i += 1
    return points, fluxes



def createGrid(point_list, xBinSize, yBinSize):
    """
    :param point_list:  array of lists with (x,y) coordinates of each center.
    :param xBinSize: x length of each rectangle in the knot grid.
    :param yBinSize: y length of each rectangle in the knot grid.
    :return:
    """
    unzip_point_list = list((zip(*point_list)))
    xmin, xmax = min(unzip_point_list[0]), max(unzip_point_list[0])
    ymin, ymax = min(unzip_point_list[1]), max(unzip_point_list[1])
    return [(x, y) for x in arange(xmin, xmax, xBinSize) for y in arange(ymin, ymax, yBinSize)]


def associateFluxes(knots, nearIndices, points, fluxes):
    """

    :param knots: array of lists with (x,y) coordinates of each vertex in the knot grid.
    :param nearIndices: array of arrays, each with the indices of the 4 nearest knots to each element in points.
    :param points: array of lists with (x,y) coordinates of each center.
    :param fluxes: array of fluxes corresponding to each element in points.
    :return:
    """
    knot_fluxes = [[] for k in knots]
    for kp in range(len(points)):
        N = nearIndices[kp][0]
        knot_fluxes[N].append(fluxes[kp])

    return [mean(fluxes) if len(fluxes) is not 0 else 0 for fluxes in knot_fluxes]

def interpolateFlux(knots, knotFluxes, points, nearIndices, xBinSize, yBinSize, normFactor):
    """
        Args:
        knots (array): array of lists with (x,y) coordinates of each vertex in the knot grid.
        knotFluxes (array): array of the flux associated with each knot.
        points (array): array of lists with (x,y) coordinates of each center.
        nearIndices (array): array of arrays, each with the indices of the 4 nearest knots to each element in points.
        xBinSize (float): x length of each rectangle in the knot grid.
        yBinSize (float): y length of each rectangle in the knot grid.
        normFactor (float): (1/xBinSize) * (1/yBinSize)

        Returns:
        array: array of interpolated flux at each point in points.

        """

    interpolated_fluxes = []

    for kp, point in enumerate(points):
        nearest_fluxes = [knotFluxes[i] for i in nearIndices[kp]]
        # If any knot has no flux, use nearest neighbor interpolation.
        if 0 in nearest_fluxes:
            N = nearIndices[kp][0]
            interpolated_fluxes.append(knotFluxes[N])
        # Else, do bilinear interpolation
        else:
            deltaX1 = abs(point[0] - knots[nearIndices[kp][0]][0])
            deltaX2 = xBinSize - deltaX1
            deltaY1 = abs(point[1] - knots[nearIndices[kp][0]][1])
            deltaY2 = yBinSize - deltaY1
            # Normalize distances with factor
            # Interpolate
            interpolated_fluxes.append(normFactor * (deltaX1 * deltaY2 * nearest_fluxes[0]
                                                     + deltaX2 * deltaY2 * nearest_fluxes[1]
                                                     + deltaX2 * deltaY1 * nearest_fluxes[2]
                                                     + deltaX1 * deltaY1 * nearest_fluxes[3]))
    return interpolated_fluxes


def nearestIndices(points, knotTree):
    """
        Args:
        points (array): array of lists with (x,y) coordinates of each center.
        knotTree (spatial.KDTree): spatial.KDTree(knots)

        Returns:
        array: array of arrays, each with the indices of the 4 nearest knots to each element in points.

        """
    return array([nearest(point, 4, knotTree) for point in points])


def BLISS(points, fluxes, knots, nearIndices, xBinSize=0.01, yBinSize=0.01, normFactor=10000):
    """
        Args:
        points (array): array of lists with (x,y) coordinates of each center.
        fluxes (array): array of fluxes corresponding to each element in points.
        knots (array): array of lists with (x,y) coordinates of each vertex in the knot grid.
        nearIndices (array): array of arrays, each with the indices of the 4 nearest knots to each element in points.
        xBinSize (float): x length of each rectangle in the knot grid.
        yBinSize (float): y length of each rectangle in the knot grid.
        normFactor (float): (1/xBinSize) * (1/yBinSize)

        Returns:
        array: array of interpolated flux at each point in points.

        """
    meanKnotFluxes = associateFluxes(knots, nearIndices, points, fluxes)
    interpolFluxes = interpolateFlux(knots=knots, knotFluxes=meanKnotFluxes, points=points,
                                     nearIndices=nearIndices, xBinSize=xBinSize, yBinSize=yBinSize,
                                     normFactor=normFactor)
    return interpolFluxes
