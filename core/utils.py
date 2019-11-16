# This script contains functions to load post-stack seismic data and acoustic_impedance model along the same slice of
# the survey

import os
from os.path import join as pjoin

import numpy as np

import segyio


def marmousi_seismic():
    """Function returns the seismic data that comes with the marmousi model"""
    seismic = segyio.cube(pjoin('data', 'Kirchhoff_PoSDM.segy'))
    seismic = np.transpose(seismic, axes=[1, 0, 2])
    return seismic


def marmousi_model():
    """Function returns the marmousi acoustic impedance model"""
    den_file = segyio.open(pjoin('data', 'MODEL_DENSITY_1.25m.segy'))
    rho = segyio.cube(den_file).squeeze().T
    rho = rho[:, ::5]
    v_file = segyio.open('./data/MODEL_P-WAVE_VELOCITY_1.25m.segy')
    vp = segyio.cube(v_file).squeeze().T
    vp = vp[:, ::5]
    AI = vp * rho
    return AI


