import BLISS as bliss
import matplotlib.pyplot as plt
from scipy import spatial 

from os import environ

def setup_BLISS_inputs_from_file(dataDir, xBinSize=0.01, yBinSize=0.01, 
                                 xSigmaRange=4, ySigmaRange=4, fSigmaRange=4):
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
        xcenters (nDarray): X positions for centering analysis
        ycenters (nDarray): Y positions for centering analysis
        fluxes (nDarray): normalized photon counts from raw data
        flux_err (nDarray): normalized photon uncertainties
        knots (nDarray): locations and initial flux values (weights) for interpolation grid
        nearIndices (nDarray): nearest neighbour indices per point for location of nearest knots
        keep_inds (list): list of indicies to keep within the thresholds set
    
    """
    times, xcenter, ycenters, fluxes, flux_errs = bliss.extractData(dataDir)
    
    keep_inds = bliss.removeOutliers(xcenters, ycenters, fluxes, xSigmaRange, ySigmaRange, fSigmaRange)
    
    knots = bliss.createGrid(xcenters[keep_inds], ycenters[keep_inds], xBinSize, yBinSize)
    knotTree = spatial.cKDTree(knots)
    nearIndices = bliss.nearestIndices(xcenters[keep_inds], ycenters[keep_inds], knotTree)
    normFactor = (1/xBinSize) * (1/yBinSize)
    
    return times, xcenters, ycenters, fluxes, flux_err, knots, nearIndices, keep_inds

dataDir = environ['HOME'] + "/Research/PlanetName/data/centers_and_flux_data.joblib.save"

times, xcenters, ycenters, fluxes, flux_err, knots, nearIndices, keep_inds = setup_BLISS_inputs_from_file(dataDir)

interpolFluxes = BLISS(xcenters[keep_inds], ycenters[keep_inds], fluxes[keep_inds], 
                       knots, nearIndices, 
                       xBinSize  = xBinSize, 
                       yBinSize  = yBinSize, 
                       normFactor= normFactor)
y,x = 0,1

def batman_wrapper_lmfit(times, period, tCenter, inc, aprs, rprs, edepth, ecc, omega, u1, u2, 
                         ldtype='quadratic', transitType='primary'):
    

  
def transit_model(model_params, times):
    # Transit Parameters
    u1      = model_params['u1'].value
    u2      = model_params['u2'].value
    
    if edepth > 0:
      delta_phase = deltaphase_eclipse(ecc, omega) if ecc is not 0.0 else 0.5
      t_secondary = tCenter + period*delta_phase
    
    rprs  = np.sqrt(model_params['tdepth'].value)
    
    bm_params           = batman.TransitParams() # object to store transit parameters
    
    bm_params.per       = model_params['period'].value   # orbital period
    bm_params.t0        = model_params['tCenter'].value  # time of inferior conjunction
    bm_params.inc       = model_params['inc'].value      # inclunaition in degrees
    bm_params.a         = model_params['aprs'].value     # semi-major axis (in units of stellar radii)
    bm_params.rp        = rprs     # planet radius (in units of stellar radii)
    bm_params.fp        = model_params['edepth'].value   # planet radius (in units of stellar radii)
    bm_params.ecc       = model_params['ecc'].value      # eccentricity
    bm_params.w         = model_params['omega'].value    # longitude of periastron (in degrees)
    bm_params.limb_dark = ldtype   # limb darkening model # NEED TO FIX THIS
    bm_params.u         = [u1, u2] # limb darkening coefficients # NEED TO FIX THIS
    
    m_eclipse = batman.TransitModel(bm_params, times, transittype=transitType)# initializes model
    
    return m_eclipse.light_curve(bm_params)

def residuals_func(model_params, times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices):
    intcpt = model_params['intcpt'] if 'intcpt' in model_params.keys() else 1.0 # default
    slope  = model_params['slope']  if 'slope'  in model_params.keys() else 0.0 # default
    crvtur = model_params['crvtur'] if 'crvtur' in model_params.keys() else 0.0 # default
    
    transit_model = transit_model(model_params, times)
    line_model    = intcpt + slope*(times-times.mean()) + crvtur*(times-times.mean())**2.
    
    # setup non-systematics model (i.e. (star + planet) / star
    model         = transit_model*line_model
    
    # multiply non-systematics model by systematics model (i.e. BLISS)
    model        *= bliss.BLISS(xcenters, ycenters fluxes/model, knots, nearIndices)
    
    return (model - fluxes) / flux_errs

initialParams = Parameters()

initialParams.add_many(
    ('period'   , init_period, False),
    ('tCenter'  , init_t0    , True , init_t0 - 0.1, init_t0 + 0.1),
    ('inc'      , init_inc   , False, 80.0, 90.),
    ('aprs'     , init_aprs  , False, 0.0, 100.),
    ('tdepth'   , init_rprs  , True , 0.0, 0.3 ),
    ('edepth'   , init_fpfs  , False, 0.0, 0.05),
    ('ecc'      , init_ecc   , False, 0.0, 1.0 ),
    ('omega'    , init_omega , False, 0.0, 1.0 ),
    ('u1'       , init_u1    , True , 0.0, 1.0 ),
    ('u2'       , init_u2    , True , 0.0, 1.0 ),
    ('intcpt'   , 1.0        , True ),
    ('slope'    , 0.0        , True ),
    ('crvtur'   , 0.0        , False))

# Reduce the number of inputs in the objective function sent to LMFIT
#   by setting the static vectors as static in the wrapper function
partial_residuals  = partial(residuals_func, 
                             times       = times,
                             xcenters    = xcenters, 
                             ycenters    = ycenters, 
                             flux        = fluxes / np.median(fluxes), 
                             fluxerr     = flux_errs / np.median(fluxes)
                             knots       = knots,
                             nearIndices = nearIndices
                             )

# Setup up the call to minimize the residuals (i.e. ChiSq)
mle0  = Minimizer(partial_residuals, initialParams)

start = time()

fitResult = mle0.leastsq() # Go-Go Gadget Fitting Routine

print("LMFIT operation took {} seconds".format(time()-start))

report_errors(fitResult.params)

fig1 = figure()
ax11= fig1.add_subplot(221)
ax12= fig1.add_subplot(222)
ax21= fig1.add_subplot(223)
ax22= fig1.add_subplot(224)

ax11.scatter(xcenters, fluxes, s=0.1, alpha=0.1)
ax12.scatter(ycenters, fluxes, s=0.1, alpha=0.1)
ax21.scatter(xcenters, ycenters, s=0.1, alpha=0.1, c=interpolFluxes)
ax22.scatter(xcenters, ycenters, s=0.1, alpha=0.1, c=(fluxes-interpolFluxes)**2)

fig2 = figure()
ax1   = fig2.add_subplot(211)
ax2   = fig2.add_subplot(212)
ax1.scatter(times, fluxes)
ax1.scatter(times, interpolFluxes)
ax2.scatter(times, fluxes - interpolFluxes)
