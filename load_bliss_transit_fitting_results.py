import argparse
import batman
import BLISS as bliss
import corner
import exoparams
import json
import matplotlib.pyplot as plt
import numpy as np

from functools import partial
from lmfit import Parameters, Minimizer, report_errors
from os import environ
from pandas import DataFrame
from scipy import spatial 
from sklearn.externals import joblib
from statsmodels.robust import scale
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
    
    keep_inds = bliss.removeOutliers(xcenters, ycenters, x_sigma_cutoff=xSigmaRange, y_sigma_cutoff=ySigmaRange)
    #, fSigmaRange)
    
    knots = bliss.createGrid(xcenters[keep_inds], ycenters[keep_inds], xBinSize, yBinSize)
    knotTree = spatial.cKDTree(knots)
    nearIndices = bliss.nearestIndices(xcenters[keep_inds], ycenters[keep_inds], knotTree)
    
    return times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds

def deltaphase_eclipse(ecc, omega):
    return 0.5*( 1 + (4. / pi) * ecc * cos(omega))

def transit_model_func(model_params, times, ldtype='quadratic', transitType='primary'):
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

# def residuals_func(model_params, times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds,
#                     xBinSize  = 0.1, yBinSize  = 0.1):
#     intcpt = model_params['intcpt'] if 'intcpt' in model_params.keys() else 1.0 # default
#     slope  = model_params['slope']  if 'slope'  in model_params.keys() else 0.0 # default
#     crvtur = model_params['crvtur'] if 'crvtur' in model_params.keys() else 0.0 # default
#
#     transit_model = transit_model_func(model_params, times[keep_inds])
#
#     line_model    = intcpt + slope*(times[keep_inds]-times[keep_inds].mean()) \
#                            + crvtur*(times[keep_inds]-times[keep_inds].mean())**2.
#
#     # setup non-systematics model (i.e. (star + planet) / star
#     model         = transit_model*line_model
#
#     # compute the systematics model (i.e. BLISS)
#     sensitivity_map = bliss.BLISS(  xcenters[keep_inds],
#                                     ycenters[keep_inds],
#                                     fluxes[keep_inds],
#                                     knots, nearIndices,
#                                     xBinSize  = xBinSize,
#                                     yBinSize  = yBinSize
#                                  )
#
#     # Identify when something very weird happens, and the sensitivity_map fails
#     #   This may be related to the outlier points that use KNN instead of interpolation
#     #   We will mitigate these outliers by replacing them as the mean of their neighbours
#     #       or the closest neighbour, in the corner cases
#     nSig = 10
#     vbad_sm = np.where(abs(sensitivity_map - np.median(sensitivity_map)) > nSig*scale.mad(sensitivity_map))[0]
#
#     # Corner Cases that Cause Faults with Average Replacement
#     if len(sensitivity_map)-1 in vbad_sm:
#         vbad_sm = np.array(list(set(vbad_sm) - set(len(sensitivity_map))))
#         end_corner_case = True
#     else:
#         end_corner_case = False
#
#     if 0 in vbad_sm:
#         vbad_sm = np.array(list(set(vbad_sm) - set([0])))
#         start_corner_case = True
#     else:
#         start_corner_case = False
#
#     # Default outlier mitigation
#     sensitivity_map[vbad_sm] = 0.5*(sensitivity_map[vbad_sm-1] + sensitivity_map[vbad_sm+1])
#
#     # Address outliers at the start and end of the array
#     #   This is equivalent to the "nearest neighbour" interpolation
#     if end_corner_case: sensitivity_map[-1] = sensitivity_map[2]
#     if start_corner_case: sensitivity_map[0] = sensitivity_map[1]
#
#     model = model * sensitivity_map
#
#     return (model - fluxes[keep_inds]) / flux_errs[keep_inds] # should this be squared?

def generate_best_fit_solution(model_params, times, xcenters, ycenters, fluxes, knots, nearIndices, keep_inds, 
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
    
    output = {}
    output['full_model'] = model
    output['line_model'] = line_model
    output['transit_model'] = transit_model
    output['bliss_map'] = sensitivity_map
    
    return output

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

ap = argparse.ArgumentParser()
ap.add_argument('-f', '--filename'      , type=str  , required=True , default=''  , help='File storing the times, xcenters, ycenters, fluxes, flux_errs')
ap.add_argument('-pn', '--planet_name'  , type=str  , required=True , default=''  , help='Either the string name of the planet from Exoplanets.org or a json file containing ')
ap.add_argument('-xb', '--xbinsize'     , type=float, required=False, default=0.1 , help='Stepsize in X-sigma to space the knots')
ap.add_argument('-yb', '--ybinsize'     , type=float, required=False, default=0.1 , help='Stepsize in Y-sigma to space the knots')
ap.add_argument('-pl', '--plot2screen'  , type=bool , required=False, default=False, help='Toggle whether to Plot to Screen or Not')
ap.add_argument('-sh', '--save_header'  , type=str  , required=False, default='rename_me_', help='Save name header to save LMFIT joblibe and plots to; set to `None` to dis saving')
ap.add_argument('-rm', '--run_mcmc_now' , type=bool , required=False, default=True, help='Toggle whether to Run LMFIT Now or just use the init values')
ap.add_argument('-rl', '--run_lmfit_now', type=bool , required=False, default=True, help='Toggle whether to Run the MCMC Now or just LMFIT/Init')

args = vars(ap.parse_args())

# dataDir = environ['HOME'] + "/Research/PlanetName/data/centers_and_flux_data.joblib.save"
dataDir       = args['filename']
xBinSize      = args['xbinsize']
yBinSize      = args['ybinsize']
planet_name   = args['planet_name']
plot_now      = args['plot2screen']
save_header   = args['save_header']
run_mcmc_now  = args['run_mcmc_now']
run_lmfit_now = args['run_lmfit_now']

init_u1, init_u2, init_u3, init_u4, init_fpfs = None, None, None, None, None

if planet_name[-5:] == '.json':
    with open(planet_name, 'r') as file_in:
        planet_json = json.load(file_in)
    init_period = planet_json['period']
    init_t0     = planet_json['t0']
    init_aprs   = planet_json['aprs']
    init_inc    = planet_json['inc']
    
    if 'tdepth' in planet_json.keys():
        init_tdepth   = planet_json['tdepth']
    elif 'rprs' in planet_json.keys():
        init_tdepth   = planet_json['rprs']**2
    elif 'rp' in planet_json.keys():
        init_tdepth   = planet_json['rp']**2
    else:
        raise ValueError("Eitehr `tdepth` or `rprs` or `rp` (in relative units) \
                            must be included in {}".format(planet_name))
    
    init_fpfs   = planet_json['fpfs'] if 'fpfs' in planet_json.keys() else 500 / ppm
    init_ecc    = planet_json['ecc']
    init_omega  = planet_json['omega']
    init_u1     = planet_json['u1'] if 'u1' in planet_json.keys() else None
    init_u2     = planet_json['u2'] if 'u2' in planet_json.keys() else None
    init_u3     = planet_json['u3'] if 'u3' in planet_json.keys() else None
    init_u4     = planet_json['u4'] if 'u4' in planet_json.keys() else None
    
    if 'planet name' in planet_json.keys():
        planet_name = planet_json['planet name']
    else:
        # Assume the json file name is the planet name
        #   This is a bad assumption; but it is one that users will understand
        print("'planet name' is not inlcude in {};".format(planet_name), end=" ")
        planet_name = planet_name.split('.json')[0]
        print(" assuming the 'planet name' is {}".format(planet_name))    
else:
    init_period, init_t0, init_aprs, init_inc, init_tdepth, init_ecc, init_omega = exoparams_to_lmfit_params(planet_name)

init_fpfs = 500 / ppm if init_fpfs is None else init_fpfs
init_u1   = 0.1 if init_u1 is None else init_u1
init_u2   = 0.0 if init_u2 is None else init_u2
init_u3   = 0.0 if init_u3 is None else init_u3
init_u4   = 0.0 if init_u4 is None else init_u4

print('Acquiring Data')
times, xcenters, ycenters, fluxes, flux_errs, knots, nearIndices, keep_inds = setup_BLISS_inputs_from_file(dataDir)

print('Fixing Time Stamps')
len_init_t0 = len(str(int(init_t0)))
len_times = len(str(int(times.mean())))

# Check if `init_t0` is in JD or MJD
if len_init_t0 == 7 and len_times != 7:
    if len_times == 5:
        init_t0 = init_t0 - 2400000.5
    elif len_times == 4:
        init_t0 = init_t0 - 2450000.5
    else:
        raise ValueError('The `init_t0` is {} and `times.mean()` is {}'.format(int(init_t0), int(times.mean())))

# Check if `init_t0` is in MJD or Simplified-MJD
if len(str(int(init_t0))) > len(str(int(times.mean()))): init_t0 = init_t0 - 50000

print('Initializing Parameters')
initialParams = Parameters()

# fluxes_std = np.std(fluxes/np.median(fluxes))

initialParams.add_many(
    ('period'   , init_period, False),
    ('tCenter'  , init_t0    , True  , init_t0 - 0.1, init_t0 + 0.1),
    ('inc'      , init_inc   , False, 80.0, 90.),
    ('aprs'     , init_aprs  , False, 0.0, 100.),
    ('tdepth'   , init_tdepth, True , 0.0, 0.3 ),
    ('edepth'   , init_fpfs  , False, 0.0, 0.05),
    ('ecc'      , init_ecc   , False, 0.0, 1.0 ),
    ('omega'    , init_omega , False, 0.0, 1.0 ),
    ('u1'       , init_u1    , True , 0.0, 1.0 ),
    ('u2'       , init_u2    , True, 0.0, 1.0 ),
    ('intcpt'   , 1.0        , True ),#, 1.0-1e-3 + 1.0+1e-3),
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

if run_lmfit_now:
    print('LM-Fitting the Model')
    # Setup up the call to minimize the residuals (i.e. ChiSq)
    mle0  = Minimizer(partial_residuals, initialParams)

    start = time()

    fitResult = mle0.leastsq() # Go-Go Gadget Fitting Routine

    print("LMFIT operation took {} seconds".format(time()-start))

    if save_header is not 'None': joblib.dump(fitResult, '{}_LMFIT_fitResults.joblib.save'.format(save_header))
    
    report_errors(fitResult.params)
else:
    fitResult.params = initialParams

print('Establishing the Best Fit Solution')
bf_model_set = generate_best_fit_solution(fitResult.params, 
                                            times, xcenters, ycenters, fluxes / np.median(fluxes), 
                                            knots, nearIndices, keep_inds, 
                                            xBinSize  = xBinSize, yBinSize  = yBinSize)

bf_full_model    = bf_model_set['full_model']
bf_line_model    = bf_model_set['line_model']
bf_transit_model = bf_model_set['transit_model']
bf_bliss_map     = bf_model_set['bliss_map']

nSig = 10
good_bf = np.where(abs(bf_full_model - np.median(bf_full_model)) < nSig*scale.mad(bf_full_model))[0]

# print('FINDME:', (fluxes[keep_inds]).mean(),
#         bf_full_model.mean(), bf_line_model.mean(), bf_transit_model.mean(), bf_bliss_map.mean())

# plt.hist(bf_full_model,bins=bf_full_model.size//10)
# plt.show()

if plot_now or save_header is not 'None':
    print('Plotting the Correlations')
    fig1 = plt.figure()
    ax11 = fig1.add_subplot(221)
    ax12 = fig1.add_subplot(222)
    ax21 = fig1.add_subplot(223)
    ax22 = fig1.add_subplot(224)

    ax11.scatter(xcenters[keep_inds][good_bf], fluxes[keep_inds][good_bf], s=0.1, alpha=0.1)
    ax12.scatter(ycenters[keep_inds][good_bf], fluxes[keep_inds][good_bf], s=0.1, alpha=0.1)
    ax21.scatter(xcenters[keep_inds][good_bf], ycenters[keep_inds][good_bf],
                    s=0.1, alpha=0.1, c=(bf_bliss_map*bf_line_model)[good_bf])
    ax22.scatter(xcenters[keep_inds][good_bf], ycenters[keep_inds][good_bf],
                    s=0.1, alpha=0.1, c=(fluxes[keep_inds]-bf_full_model)[good_bf]**2)

    ax11.set_title('Xcenters vs Normalized Flux')
    ax21.set_title('Ycenters vs Normalized Flux')
    ax12.set_title('X\&Y Centers vs Bliss Map')
    ax22.set_title('X\&Y Centers vs Residuals (Flux - Bliss Map)')

    nSig = 3
    xCtr = xcenters[keep_inds][good_bf].mean()
    xSig = xcenters[keep_inds][good_bf].std()

    yCtr = ycenters[keep_inds][good_bf].mean()
    ySig = ycenters[keep_inds][good_bf].std()

    ax11.set_xlim(xCtr - nSig * xSig, xCtr + nSig * xSig)
    ax12.set_xlim(yCtr - nSig * ySig, yCtr + nSig * ySig)
    ax21.set_xlim(xCtr - nSig * xSig, xCtr + nSig * xSig)
    ax21.set_ylim(yCtr - nSig * ySig, yCtr + nSig * ySig)
    ax22.set_xlim(xCtr - nSig * xSig, xCtr + nSig * xSig)
    ax22.set_ylim(yCtr - nSig * ySig, yCtr + nSig * ySig)

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # plt.tight_layout()

    print('Plotting the Time Series')

    fig2 = plt.figure()
    ax1 = fig2.add_subplot(211)
    ax2 = fig2.add_subplot(212)
    ax1.scatter(times[keep_inds][good_bf], fluxes[keep_inds][good_bf] , s=0.1, alpha=0.1)
    ax1.scatter(times[keep_inds][good_bf], bf_full_model[good_bf], s=0.1, alpha=0.1)
    ax2.scatter(times[keep_inds][good_bf], (fluxes[keep_inds] - bf_full_model)[good_bf], s=0.1, alpha=0.1)

    ax1.set_title('{} Raw CH2 Light Curve with BLISS + Linear + BATMAN Model'.format(planet_name))
    ax2.set_title('{} Raw CH2 Residuals (blue - orange above)'.format(planet_name))

    mng = plt.get_current_fig_manager()
    mng.window.showMaximized()
    # plt.tight_layout()
    
    if plot_now: plt.show()
    
    if save_header is not 'None': 
        fig1.savefig('{}_fig1_BLISS_Correlations_Plots_with_LMFIT.png'.format(save_header))
        fig2.savefig('{}_fig2_BLISS_Time_Series_Fits_and_Residuals_with_LMFIT.png'.format(save_header))

if run_mcmc_now:
    print("MCMC Sampling the Posterior Space")
    
    mle0.params.add('f', value=1, min=0.001, max=2)
    
    def logprior_func(p):
        return 0
    
    def lnprob(p):
        logprior = logprior_func(p)
        if not np.isfinite(logprior):
            return -np.inf
        
        resid = partial_residuals(p)
        s = p['f']
        resid *= 1 / s
        resid *= resid
        resid += np.log(2 * np.pi * s**2)
        return -0.5 * np.sum(resid) + logprior
    
    mini  = Minimizer(lnprob, mle0.params)
    
    start = time()
    
    #import emcee
    #res = emcee.sampler(lnlikelihood = lnprob, lnprior=logprior_func)

    res   = mini.emcee(params=mle0.params, steps=100, nwalkers=100, burn=1, thin=10, ntemps=1,
                        pos=None, reuse_sampler=False, workers=1, float_behavior='posterior',
                        is_weighted=True, seed=None)
    
    print("MCMC operation took {} seconds".format(time()-start))
    emcee_save_name = save_header + 'emcee_sample_results.joblib.save'
    print("Saving EMCEE results to {}".format(emcee_save_name))
    joblib.dump(res,emcee_save_name)
    
    res_var_names = np.array(res.var_names)
    res_flatchain = np.array(res.flatchain)
    res_df = DataFrame(res_flatchain, columns=res_var_names)
    res_df = res_df.drop(['u2','slope'], axis=1)
    print(res_df)
    
    corner_kw = dict(levels=[0.68, 0.95, 0.997], plot_datapoints=False, 
                        smooth=True, smooth1d=True, bins=100, 
                        range=[(54945,54990),(0.01357,0.01385),(0.1097,0.11035),\
                                    (0.996,1.002), (0.998,1.003)], 
                        plot_density=False, fill_contours=True, color='darkblue')
    
    corner.corner(res_df, **corner_kw)
    
    corner_save_name = save_header + 'mcmc_corner_plot.png'
    print('Saving MCMC Corner Plot to {}'.format(corner_save_name))
    plt.savefig(corner_save_name)