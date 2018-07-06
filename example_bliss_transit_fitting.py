import BLISS
import matplotlib.pyplot as plt

from lmfit import Minimizer, Parameters
from os import environ
from scipy import spatial 

def setup_BLISS_inputs_from_file(dataDir, xBinSize=0.01, yBinSize=0.01, xSigmaRange=3, ySigmaRange=3):
    """This function takes in the filename of the data (stored with sklearn-joblib),
        checks the data for outliers, establishes the interpolation grid, 
        computes the nearest neighbours between all data points and that grid, 
        and outputs the necessary values for using BLISS
        
        The `flux` is assumed to be pure stellar signal -- i.e. no planet. 
        BLISS is expected to be used inside a fitting routine where the transit has been `divided out`.
        This example here assumes that there is no transit or eclipse in the light curve data (i.e. `flux` == 'stellar flux').
        To use this with a light curve that contains a transit or eclipse, send the "residuals" to BLISS:
            - i.e. `flux = system_flux / transit_model`
    
    Written by C.Munoz 07-05-18
    Edited by J.Fraine 07-06-18

    Args:
        dataDir (str): the directory location for the joblib file containing the x,y,flux information
        
        xBinSize (float): distance in x-dimension to space interpolation grid
        yBinSize (float): distance in y-dimension to space interpolation grid
        xSigmaRange (float): relative distance in gaussian sigma space to reject x-outliers
        ySigmaRange (float): relative distance in gaussian sigma space to reject y-outliers

    Returns:
        points (nDarray): X and Y positions for centering analysis
        fluxes (nDarray): normalized photon counts from raw data
        knots (nDarray): locations and initial flux values (weights) for interpolation grid
        nearIndices (nDarray): nearest neighbour indices per point for location of nearest knots
    
    """
    times, points, fluxes, flux_errs = BLISS.extractData(dataDir)
    points, fluxes = BLISS.removeOutliers(points, fluxes, xSigmaRange, ySigmaRange)
    knots = BLISS.createGrid(points, xBinSize, yBinSize)
    knotTree = spatial.cKDTree(knots)
    nearIndices = BLISS.nearestIndices(points, knotTree)
    normFactor = (1/xBinSize) * (1/yBinSize)
    
    return times, points, fluxes, flux_errs, knots, nearIndices

dataDir = environ['HOME'] + "/Research/PlanetName/data/centers_and_flux_data.joblib.save"

points, fluxes, knots, nearIndices = setup_BLISS_inputs_from_file(dataDir)

interpolFluxes = BLISS.BLISS(points, fluxes, knots, nearIndices, 
                       xBinSize=xBinSize, yBinSize = yBinSize, 
                       normFactor=normFactor)

if not isinstance(points, np.ndarray): points = np.array(points)

y,x = 0,1

def residuals_func(model_params, times, flux, fluxerr):
    model = transit_line_model(model_params, times)
    return (model - flux) / fluxerr

partial_residualspartial_  = partial(residuals_func, 
                            times  = timeSliceKmod, 
                            flux   = fluxSliceK / np.median(fluxSliceK), 
                            fluxerr= ferrSliceK / np.median(fluxSliceK)
                            )

mle0  = Minimizer(partial_residuals, initialParams)

start = time()
fitResult = mle0.leastsq()
print("LMFIT operation took {} seconds".format(time()-start))

report_errors(fitResult.params)

# # plt.scatter([p[0] for p in points], [p[1] for p in points], s=0.1, c=interpolFluxes)
# plt.scatter(points.T[x], points.T[y], s=0.1, c=interpolFluxes)
# plt.colorbar()
# plt.show()
