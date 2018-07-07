# EXAMPLE USAGE: python example_bliss_transit_fitting.py -f ../../data/group0_gsc.joblib.save -p 'GJ 1214 b'

import argparse
import batman
import BLISS as bliss
import exoparams
import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from lmfit import Parameters, Minimizer
from os import environ
from scipy import spatial 
from time import time

y,x = 0,1
ppm = 1e6

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
    times, xcenters, ycenters, fluxes, flux_errs = bliss.extractData(dataDir)
    
    keep_inds = bliss.removeOutliers(xcenters, ycenters, fluxes, xSigmaRange, ySigmaRange, fSigmaRange)
    
    knots = bliss.createGrid(xcenters[keep_inds], ycenters[keep_inds], xBinSize, yBinSize)
    knotTree = spatial.cKDTree(knots)
    nearIndices = bliss.nearestIndices(xcenters[keep_inds], ycenters[keep_inds], knotTree)
    
    return times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds


def deltaphase_eclipse(ecc, omega):
    return 0.5*( 1 + (4. / pi) * ecc * cos(omega))

def transit_model_func(model_params, times, ldtype='quadratic', transittype='primary'):
    # Transit Parameters
    u1      = model_params['u1'].value
    u2      = model_params['u2'].value
    
    if 'edepth' in model_params.keys() and model_params['edepth'] > 0:
        if 'ecc' in model_params.keys() and 'omega' in model_params.keys() and model_params['ecc'] > 0:
            delta_phase = deltaphase_eclipse(model_params['ecc'], model_params['omega'])
        else:
            delta_phase = 0.5
        
        t_secondary = model_params['tCenter'] + model_params['period']*delta_phase
        
    else:
        model_params.add('edepth', 0.0, False)
    
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

def residuals_func(model_params, times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds, 
                    xBinSize  = 0.1, yBinSize  = 0.1):
    intcpt = model_params['intcpt'] if 'intcpt' in model_params.keys() else 1.0 # default
    slope  = model_params['slope']  if 'slope'  in model_params.keys() else 0.0 # default
    crvtur = model_params['crvtur'] if 'crvtur' in model_params.keys() else 0.0 # default
    
    transit_model = transit_model_func(model_params, times[keep_inds])
    
    line_model    = intcpt + slope*(times[keep_inds]-times[keep_inds].mean()) \
                           + crvtur*(times[keep_inds]-times[keep_inds].mean())**2.
    
    # setup non-systematics model (i.e. (star + planet) / star
    model         = transit_model*line_model
    
    # compute the systematics model (i.e. BLISS)
    sensitivity_map = bliss.BLISS(  xcenters[keep_inds], 
                                    ycenters[keep_inds], 
                                    fluxes[keep_inds], 
                                    knots, nearIndices, 
                                    xBinSize  = xBinSize, 
                                    yBinSize  = yBinSize
                                 )
    
    model = model * sensitivity_map
    
    return (model - fluxes[keep_inds]) / flux_errs[keep_inds] # should this be squared?

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filename'   , type=str  , required=True , default='' , help='File storing the times, xcenters, ycenters, fluxes, flux_errs')
ap.add_argument('-p', '--planet_name', type=str  , required=True , default='' , help='Either the string name of the planet from Exoplanets.org or a json file containing ')
ap.add_argument('-xb', '--xbinsize'  , type=float, required=False, default=0.1, help='Stepsize in X-sigma to space the knots')
ap.add_argument('-yb', '--ybinsize'  , type=float, required=False, default=0.1, help='Stepsize in Y-sigma to space the knots')
args = vars(ap.parse_args())

# dataDir = environ['HOME'] + "/Research/PlanetName/data/centers_and_flux_data.joblib.save"
dataDir     = args['filename']
xBinSize    = float(args['xbinsize'])
yBinSize    = float(args['ybinsize'])
planet_name = args['planet_name']

def exoparams_to_lmfit_params(planet_name):
    ep_params   = exoparams.PlanetParams(planet_name)
    iApRs       = ep_params.ar.value
    iEcc        = ep_params.ecc.value
    iInc        = ep_params.i.value
    iPeriod     = ep_params.per.value
    iTCenter    = ep_params.tt.value
    iTdepth     = ep_params.depth.value
    iOmega      = ep_params.om.value
    
    return iPeriod, iTCenter, iApRs, iInc, iTdepth, iEcc, iOmega

if planet_name[:-5] == '.json':
    with open(planet_name, 'r') as file_in:
        planet_json = json.load(file_in)
    init_period = planet_json['period']
    init_t0     = planet_json['t0']
    init_aprs   = planet_json['aprs']
    init_inc    = planet_json['inc']
    init_tdepth = planet_json['tdepth']
    init_ecc    = planet_json['ecc']
    init_omega  = planet_json['omega']
else:
    init_period, init_t0, init_aprs, init_inc, init_tdepth, init_ecc, init_omega = exoparams_to_lmfit_params(planet_name)

init_fpfs        = 500 / ppm
init_u1, init_u2 = 0.1, 0.1

times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds = setup_BLISS_inputs_from_file(dataDir)

initialParams = Parameters()

initialParams.add_many(
    ('period'   , init_period, False),
    ('tCenter'  , init_t0    , True , init_t0 - 0.1, init_t0 + 0.1),
    ('inc'      , init_inc   , False, 80.0, 90.),
    ('aprs'     , init_aprs  , False, 0.0, 100.),
    ('tdepth'   , init_tdepth, True , 0.0, 0.3 ),
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
                             fluxes      = fluxes / np.median(fluxes), 
                             flux_errs   = flux_errs / np.median(fluxes),
                             knots       = knots,
                             nearIndices = nearIndices,
                             keep_inds   = keep_inds
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
