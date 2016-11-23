#!/usr/local/bin/env python

import json
import si_constants
import numpy as np
import scipy.interpolate as interp
from moby2.tod import filters
from scipy import signal

import moby2

class IdealVPM(object):
    ''' The VPM object contains the gemetric state of a VPM system and includes
    functionality to calculate the transfer function of the VPM-detector system.

    The object data are parameters of the VPM only.
    The only dynamic variable is the distance between the grid-wires and the mirror.
    All other grid parameters are static, but can be accessed/set using get/set methods.

    The ideal grid assumes the VPM consists of an ideal polarizer backed by an ideal mirror.

    If the VPM is not ideal, the caculated transfer fuction assumes the grid wires are copper plated
    tungsten wires using a thin nickel flash, and assumes the mirror is aluminum.

    '''

    def __init__(self, recv = 'q_band', fname = 'q_vpm.txt'):
        return

    def det_vpm(self, alpha, phi, theta, dists, wavelengths, weights, Is, Qs, Us, Vs):
        ''' follows the spirit and coordinate conventions of Chuss et al 2012. Non-modulated terms
        have been dropped.

        alpha (float):[radians] angle of the detectors w.r.t projected grid wires
        phi (float): [radians] angle of grid wires w.r.t. plane of incidence
        theta (float): [radians] angle of incidence
        dist (array like): [m] grid mirror separation. Note that tod.vpm is in mm. You are responsible
        for unit conversion.
        wavelengths (array like): [m]
        weights (array like): weight of respective frequency relative to unity
        '''
        num_waves = len(wavelengths)

        delays = 2.0 * dists * np.cos(theta)
        wave_nums = 2.0 * np.pi/ wavelengths
        delays = delays[:,np.newaxis]
        wave_nums = wave_nums[np.newaxis,:]
        config = delays * wave_nums

        ang_dif_factor = 1./2. * np.sin(2. * (alpha - phi))

        if Qs or Us != 0:
            c_config = np.sum(np.cos(config), axis = 1)/ num_waves
        m = 0
        if Qs != 0: # Q modulation
            m += c_config * -1. * Qs * ang_dif_factor * np.sin(2. * phi)
        if Us != 0: # U modulation
            m += c_config * -1./4. * Us * (np.sin(2. * (alpha - 2. * phi)) + np.sin(2. * alpha))
        if Vs != 0:# V modulation
            s_config = np.sum(np.sin(config), axis = 1)/ num_waves
            m += s_config * Vs * ang_dif_factor
        return m

    def slow_vpm(self, alpha, phi, theta, dists, wavelengths, weights, Is, Qs, Us, Vs):
        ''' follows the spirit and coordinate conventions of Chuss et al 2012. DC terms are droppded.
        alpha (float):[radians] angle of the detectors w.r.t projected grid wires
        phi (float): [radians] angle of grid wires w.r.t. plane of incidence [radians]
        theta (float): [radians] angle of incidence [radians]
        dist (array like): grid mirror separation [m]
        wavelengths (array like): [m]
        weights (array like): weight of respective frequency relative to unity
        '''

        delays = 2.0 * dists * np.cos(theta)
        wave_nums = 2.0 * np.pi/ wavelengths

        # the idea of this method is to take the geometry of the VPM detecotr system and input IQUV
        # and return the band averaged response measured by a detector. 

        det = np.zeros(len(delays))

        ang_dif_factor = 1./2. * np.sin(2. * (alpha - phi))
        for delay_count in range(len(delays)):
            mq = Qs * (-1. * ang_dif_factor * np.cos(delays[delay_count] * wave_nums) * np.sin(2 * phi))
            mu = -1.*Us/4. * (np.sin(2 * (alpha - 2 * phi)) + np.sin(2 * alpha)) * np.cos( delays[delay_count] * wave_nums)
            mv = Vs * ang_dif_factor * np.sin(delays[delay_count] * wave_nums)
            det_val = (mq + mu + mv) * weights
            det[delay_count] = np.sum(det_val)/len(wave_nums)
        return det
