#!/usr/bin/env python
# -*- coding: utf-8 -*-

# ---- From MG ---- #
# -- first on March, 2022 ---- #
# This is the code I received from Sean Johnson on 3/16
# doublet line ratios added; the name was Valentino2017_Strom.py
# -- update based on empirical line ratios in the literature
# based on email from Sean, in May, 2022
# -- Then Joel updated extinction in June 2022: uses the Prospector
# dust model instead. Instead of inputting A_V_stellar = 1.086 * dust2,
# now we’ll input dust_index, dust1, and dust2.
# ------------------ #
# From original code: The following come close do not exactly match the Valentino 2017
# (http://adsabs.harvard.edu/abs/2017MNRAS.472.4878V) predictions,
# probably because of use of slightly different extinction laws.


import numpy as np
import extinction
import math, random

__all__ = ['predict_L_Hemis',
           'predict_L_OII_tot', 'predict_L_OIII5007',
           'predict_L_NeIII3870', 'predict_L_NII6585',
           'predict_L_SII_tot', 'find_nearest']

# ---------------------------------------------------------------- #
def charlot_and_fall_extinction(lam, dust1, dust2, dust2_index,
                                dust1_index=-1.0,kriek=True):
    """
    returns F(obs) / F(emitted) for a given attenuation curve (dust_index) + dust1 + dust2
    """

    dust1_ext = np.exp(-dust1*(lam/5500.)**dust1_index)
    dust2_ext = np.exp(-dust2*(lam/5500.)**dust2_index)

    # -- sanitize inputs
    lam = np.atleast_1d(lam).astype(float)

    # -- are we using Kriek & Conroy 13?
    if kriek:
        dd63 = 6300.00
        lamv = 5500.0
        dlam = 350.0
        lamuvb = 2175.0

        # -- Calzetti curve, below 6300 Angstroms, else no addition
        cal00 = np.zeros_like(lam)
        gt_dd63 = lam > dd63
        le_dd63 = ~gt_dd63
        if np.sum(gt_dd63) > 0:
            cal00[gt_dd63] = 1.17*( -1.857+1.04*(1e4/lam[gt_dd63]) ) + 1.78
        if np.sum(le_dd63) > 0:
            cal00[le_dd63]  = 1.17*(-2.156+1.509*(1e4/lam[le_dd63])-0.198*(1E4/lam[le_dd63])**2 + 0.011*(1E4/lam[le_dd63])**3) + 1.78
        cal00 = cal00/0.44/4.05

        eb = 0.85 - 1.9 * dust2_index  #KC13 Eqn 3

        # -- Drude profile for 2175A bump
        drude = eb*(lam*dlam)**2 / ( (lam**2-lamuvb**2)**2 + (lam*dlam)**2 )

        attn_curve = dust2*(cal00+drude/4.05)*(lam/lamv)**dust2_index
        dust2_ext = np.exp(-attn_curve)

    ext_tot = dust2_ext*dust1_ext

    return ext_tot


# ---------------------------------------------------------------- #
def find_nearest(array,value):
    """
    - much faster than argmin
    https://stackoverflow.com/questions/2566412/find-nearest-value-in-numpy-array
    """
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

# ---------------------------------------------------------------- #
def predict_L_Hemis(logSFR, n, dust1, dust2):
    """
    Predict the Halpha to Hzeta luminosity given logSFR/(Msun/yr) and stellar Av.
    Reproduces Valentino 2017 catalog a bias of 0.005 dex
    and standard deviation of 0.007 dex.
    return: log of the Halpha luminosity in erg/s

    - older version:
        - f = 0.76 # stellar to nebular Av ratio.
        - Av_nebular = Av_stellar/f
        - ext_mags = extinction.calzetti00(out['wave'], Av_nebular, 4.05)
    """
    wavelengths = np.array([6564.61, 4862.68, 4341.68, 
                            4102.89, 3971.19, 3890.17])
    #caseB_ratios = np.array([1.0, 2.86, 6.13, 8.91, 11.51, 14.23])  # Case B recombination coefficients
    caseB_ratios = np.array([1.0, 2.86, 2.86/0.466, 
                             2.86/0.256, 2.86/0.158, 2.86/0.105])
    # Calculate the extinction for each wavelength
    ext_arr = np.array([charlot_and_fall_extinction(i_, dust1, dust2, n) for i_ in wavelengths]).flatten()


    # Calculate the intrinsic line luminosities
    logL_int = logSFR - np.log10(7.9e-42) - np.log10(caseB_ratios)

    # Apply extinction to the intrinsic line luminosities
    logL_ext = logL_int + np.log10(ext_arr)

    return logL_ext

# ---------------------------------------------------------------- #
"""
def predict_L_Hemis_old(logSFR, n, dust1, dust2):

    wavelengths = np.array([6564.61, 4862.68, 4341.68, 
                            4102.89, 3971.19, 3890.17])
    
    out = {'wave': wavelengths}
    # Calculate the extinction for each wavelength
    ext_arr = np.array([charlot_and_fall_extinction(i_, dust1, dust2, n) for i_ in wavelengths]).flatten()

    # -- Halpha -- #
    logL_Ha_int = logSFR - math.log10(7.9e-42) # K98
    logL_Ha = logL_Ha_int + math.log10(ext_arr[0])

    # -- Hbeta -- #
    logL_Hb_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) # K98 and Case B
    logL_Hb = logL_Hb_int + math.log10(ext_arr[1])

    # -- Hgamma -- K98 and Case B #
    logL_Hg_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) + math.log10(0.466)
    logL_Hg = logL_Hg_int + math.log10(ext_arr[2])

    # -- Hdelta -- K98 and Case B #
    logL_Hd_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) + math.log10(0.256)
    logL_Hd = logL_Hd_int + math.log10(ext_arr[3])

    # -- Hepsilon luminosity given logSFR/(Msun/yr) and stellar Av.
    #return: log of the Hdelta luminosity in erg/s
    # K98 and Case B
    logL_He_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) + math.log10(0.158)
    logL_He = logL_He_int + math.log10(ext_arr[4])

    # -- Predict the Hzeta luminosity given logSFR/(Msun/yr) and stellar Av.
    # -- return: log of the Hdelta luminosity in erg/s
    # K98 and Case B
    logL_Hz_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) + math.log10(0.105)
    logL_Hz = logL_Hz_int + math.log10(ext_arr[5])

    out['logL'] = np.array([logL_Ha, logL_Hb, logL_Hg, logL_Hd, logL_He, logL_Hz])
    return out['logL']
"""


# ---------------------------------------------------------------- #
def predict_L_OII_tot(logSFR, n, dust1, dust2):
    """
    Predict the [O II] luminosity given logSFR/(Msun/yr) and stellar Av.
    Reproduces Valentino 2017 catalog a bias of 0.058 dex
    and standard deviation of 0.043 dex.
    return: log of the [O II] luminosity in erg/s
    """
    #f = 0.76 # stellar to nebular Av ratio.
    logL_OII_int = logSFR - math.log10(7.9e-42) # K04 actually just K98
    #Av_nebular = Av_stellar/f
    #ext_mags = extinction.calzetti00(np.array([3728.0]), Av_nebular, 4.05)
    ext = charlot_and_fall_extinction(np.array([3728.0]), dust1, dust2, n)
    logL_OII = logL_OII_int + math.log10(ext)

    # -- 3729/3726=1.5(low density limit), 0.35(high density limit)
    # -- https://www.ucolick.org/~simard/phd/root/node21.html
    #lineratio = np.random.uniform(0.35, 1.5)
    # -- Sanders et al. 2016
    lineratio = random.uniform(0.3839, 1.4558)
    logL_OII_3729 = math.log10(10**logL_OII*(lineratio/(1.+lineratio)))
    logL_OII_3726 = math.log10(10**logL_OII*(1./(1.+lineratio)))

    return logL_OII, logL_OII_3729, logL_OII_3726


# ---------------------------------------------------------------- #
def predict_L_OIII5007(logSFR, n, dust1, dust2, logMstar):
    """
    Predict the [O III] luminosity given logSFR/(Msun/yr) and stellar Av.
    Reproduces Valentino 2017 catalog a bias of 0.025 dex
    and standard deviation of 0.019 dex.
    return: log of the [O III] luminosity in erg/s
    """
    f = 0.76 # stellar to nebular Av ratio.
    logL_Hb = predict_L_Hb(logSFR, n, dust1, dust2)
    logOIII_Hbeta = 0.3 + 0.48*math.atan(-(logMstar - 10.28))

    logL_OIII = logL_Hb + logOIII_Hbeta

    # -- OIII5007/OIII4959 = 2.98 (Storey and Zeippen 2000)
    # -- https://academic.oup.com/mnras/article/312/4/813/1017382
    lineratio = 2.98
    logL_OIII_4959 = math.log10(10**logL_OIII/lineratio)

    return logL_OIII, logL_OIII_4959


# ---------------------------------------------------------------- #
def predict_L_NeIII3870(logSFR, n, dust1, dust2, logMstar):
    """
    Predict the [Ne III] 3870 luminosity given logSFR/(Msun/yr) and stellar Av and stellar mass.
    return: log of the [Ne III] luminosity in erg/s
    """
    # f = 0.76 # stellar to nebular Av ratio.
    # Av_nebular = Av_stellar/f

    # Get Hbeta assuming no dust
    logL_Hb_int = predict_L_Hb(logSFR, 0.0, 0.0, 0.0)

    # Get [O III] assuming no dust
    logOIII_Hbeta = 0.3 + 0.48*math.atan(-(logMstar - 10.28))
    logL_OIII_int = logL_Hb_int + logOIII_Hbeta

    # Assuming [Ne III] is 10x weaker than [O III]
    logL_NeIII_int = logL_OIII_int - 1.0

    # Apply dust
    #ext_mags = extinction.calzetti00(np.array([3869.85]), Av_nebular, 4.05)
    ext = charlot_and_fall_extinction(np.array([3869.85]), dust1, dust2, n)

    logL_NeIII = logL_NeIII_int + math.log10(ext)

    return logL_NeIII


# ---------------------------------------------------------------- #
def predict_L_NII6585(logSFR, n, dust1, dust2, logMstar):
    """Best fit relation from Strom+2017"""

    logL_Ha = predict_L_Ha(logSFR, n, dust1, dust2)
    logL_Hb = predict_L_Hb(logSFR, n, dust1, dust2)
    logL_OIII = predict_L_OIII5007(logSFR, n, dust1, dust2, logMstar)[0]

    logL_NII  = 0.61/((logL_OIII - logL_Hb) - 1.12) + 0.22 + logL_Ha

    # -- NII6583/NII6548 = 3.05 (Storey and Zeippen 2000)
    # -- https://academic.oup.com/mnras/article/312/4/813/1017382
    lineratio = 3.05
    logL_NII_6548 = math.log10(10**logL_NII/lineratio)

    return logL_NII, logL_NII_6548


# ---------------------------------------------------------------- #
def predict_L_SII_tot(logSFR, n, dust1, dust2, logMstar):
    """Best fit relation from Strom+2017"""

    logL_Ha = predict_L_Ha(logSFR, n, dust1, dust2)
    logL_Hb = predict_L_Hb(logSFR, n, dust1, dust2)
    logL_OIII = predict_L_OIII5007(logSFR, n, dust1, dust2, logMstar)[0]

    logL_SII  = 0.72/((logL_OIII - logL_Hb) - 1.15) + 0.53 + logL_Ha

    # -- Sanders et al. 2016, Table 1
    lineratio = random.uniform( 0.4375, 1.4484)
    logL_SII_6716 = math.log10(10**logL_SII*(lineratio/(1.+lineratio)))
    logL_SII_6731 = math.log10(10**logL_SII*(1./(1.+lineratio)))

    return logL_SII, logL_SII_6716, logL_SII_6731


# ---------------------------------------------------------------- #
def predict_L_Ha(logSFR, n, dust1, dust2):
    """
    Predict the Halpha luminosity given logSFR/(Msun/yr) and stellar Av.
    Reproduces Valentino 2017 catalog a bias of 0.005 dex
    and standard deviation of 0.007 dex.
    return: log of the Halpha luminosity in erg/s

    - new parameters with Joel's update
        dust1:
        dust2:
        n: dust2_index
    """
    # Set stellar vs nebular extinction f
    #f = 0.76 # stellar to nebular Av ratio.

    logL_Ha_int = logSFR - math.log10(7.9e-42) # K98

    # -- previously the code uses
    # """
    # Av_nebular = Av_stellar/f
    # ext_mags = extinction.calzetti00(np.array([6564.61]), Av_nebular, 4.05)
    # logL_Ha = logL_Ha_int - math.log10(2.512**ext_mags[0])
    # """
    # -- Joel update it in June 2022 with charlot_and_fall_extinction

    ext = charlot_and_fall_extinction(np.array([6564.61]), dust1, dust2, n)
    logL_Ha = logL_Ha_int + math.log10(ext)

    return logL_Ha


# ---------------------------------------------------------------- #
def predict_L_Hb(logSFR, n, dust1, dust2):
    """
    Predict the Hbeta luminosity given logSFR/(Msun/yr) and stellar Av.
    Reproduces Valentino 2017 catalog a bias of 0.024 dex
    and standard deviation of 0.026 dex.
    return: log of the Hbeta luminosity in erg/s
    """

    logL_Hb_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) # K98 and Case B

    # """
    #f = 0.76
    #Av_nebular = Av_stellar/f
    #ext_mags = extinction.calzetti00(np.array([4862.68]), Av_nebular, 4.05)
    # """

    ext = charlot_and_fall_extinction(np.array([4862.68]), dust1, dust2, n)
    #logL_Hb = logL_Hb_int - math.log10(2.512**ext_mags[0])
    logL_Hb = logL_Hb_int + math.log10(ext)

    return logL_Hb


# ---------------------------------------------------------------- #
def predict_L_Hg(logSFR, n, dust1, dust2):
    """
    Predict the Hgamma luminosity given logSFR/(Msun/yr) and stellar Av.
    return: log of the Hgamma luminosity in erg/s
    """

    # K98 and Case B
    #f = 0.76

    logL_Hg_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) + math.log10(0.466)
    # """
    #Av_nebular = Av_stellar/f
    #ext_mags = extinction.calzetti00(np.array([4341.68]), Av_nebular, 4.05)
    #logL_Hg = logL_Hg_int - math.log10(2.512**ext_mags[0])
    # """

    ext = charlot_and_fall_extinction(np.array([4341.68]), dust1, dust2, n)
    logL_Hg = logL_Hg_int + math.log10(ext)

    return logL_Hg



# ---------------------------------------------------------------- #
def predict_L_Hd(logSFR, n, dust1, dust2):
    """
    Predict the Hdelta luminosity given logSFR/(Msun/yr) and stellar Av.
    return: log of the Hdelta luminosity in erg/s
    """

    # K98 and Case B
    logL_Hd_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) + math.log10(0.256)

    # """
    # f = 0.76
    # Av_nebular = Av_stellar/f
    # ext_mags = extinction.calzetti00(np.array([4102.89]), Av_nebular, 4.05)
    # """

    ext = charlot_and_fall_extinction(np.array([4102.89]), dust1, dust2, n)
    logL_Hd = logL_Hd_int + math.log10(ext)

    return logL_Hd


# ---------------------------------------------------------------- #
def predict_L_He(logSFR, n, dust1, dust2):
    """
    Predict the Hepsilon luminosity given logSFR/(Msun/yr) and stellar Av.
    return: log of the Hdelta luminosity in erg/s
    """
    # K98 and Case B
    # f = 0.76
    # logL_He_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) + math.log10(0.158)
    # Av_nebular = Av_stellar/f
    # ext_mags = extinction.calzetti00(np.array([3971.19]), Av_nebular, 4.05)

    logL_He_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) + math.log10(0.158)
    ext = charlot_and_fall_extinction(np.array([3971.19]), dust1, dust2, n)
    logL_He = logL_He_int + math.log10(ext)

    return logL_He


# ---------------------------------------------------------------- #
def predict_L_Hz(logSFR, n, dust1, dust2):
    """
    Predict the Hzeta luminosity given logSFR/(Msun/yr) and stellar Av.
    return: log of the Hdelta luminosity in erg/s
    """

    # # K98 and Case B
    # f = 0.76
    #
    # Av_nebular = Av_stellar/f
    # ext_mags = extinction.calzetti00(np.array([3890.17]), Av_nebular, 4.05)

    logL_Hz_int = logSFR - math.log10(7.9e-42) - math.log10(2.86) + math.log10(0.105)
    ext = charlot_and_fall_extinction(np.array([3890.17]), dust1, dust2, n)
    logL_Hz = logL_Hz_int + math.log10(ext)

    return logL_Hz

