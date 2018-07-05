import BLISS as BLISS
import matplotlib.pyplot as plt
from scipy import spatial 

y,x = 0,1
dataDir = "/Users/cmunoz/Desktop/Research/GJ1214/data/group0_gsc.joblib.save"
xBinSize = 0.01
yBinSize = 0.01
xSigmaRange = 3
ySigmaRange = 3

points, fluxes = BLISS.extractData(dataDir)
points, fluxes = BLISS.removeOutliers(points, fluxes, xSigmaRange, ySigmaRange)
knots = BLISS.createGrid(points, xBinSize, yBinSize)
knotTree = spatial.KDTree(knots)
nearIndices = BLISS.nearestIndices(points, knotTree)
normFactor = (1/xBinSize) * (1/yBinSize)

interpolFluxes = BLISS.BLISS(points, fluxes, knots, nearIndices)

plt.scatter([p[0] for p in points], [p[1] for p in points], s=0.1, c=interpolFluxes)
plt.colorbar()
plt.show()
