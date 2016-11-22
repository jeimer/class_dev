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

        #state parameter
        self._dist = 0 #[m] distance between the wires and the mirror
        self._dist_offset = 0 #[m] offset between wires-mirror distance calibration


        #user parameters
        #self._weights =  # relative weight for each wavelength.

    def set_dist_offset(self, offset):
        '''set the grid mirrorf separation offset [m]'''
        self._dist_offset = offset

    def get_dist_offset(self):
        '''return the current grid mirror separation offset [m]'''
        return self._dist_offset

    def set_dist(self, dist):
        '''set the grid mirror separtation distance [m]'''
        self._dist = dist

    def get_dist(self):
        '''return the distance between the VPM grid wires and the mirror'''
        return self._dist

    def det_vpm(self, alpha, phi, theta, dists, wavelengths, weights, Is, Qs, Us, Vs):
        ''' follows the spirit and coordinate conventions of Chuss et al 2012. DC terms are droppded.
        alpha (float):[radians] angle of the detectors w.r.t projected grid wires
        phi (float): [radians] angle of grid wires w.r.t. plane of incidence [radians]
        theta (float): [radians] angle of incidence [radians]
        dist (array like): grid mirror separation [m]
        wavelengths (array like): [m]
        weights (array like): weight of respective frequency relative to unity
        '''

        delays = 2.0 * (dists + self._dist_offset) * np.cos(theta)
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

    def vpm_mat(self, alpha, phi, theta, dists, wavelengths, weights, Is, Qs, Us, Vs, ideal = True):
        ''' follows the spirit and coordinate conventions of Chuss et al 2012. DC terms are droppded.
        alpha (float):[radians] angle of the detectors w.r.t projected grid wires
        phi (float): [radians] angle of grid wires w.r.t. plane of incidence [radians]
        theta (float): [radians] angle of incidence [radians]
        dist (array like): grid mirror separation [m]
        wavelengths (array like): [m]
        weights (array like): weight of respective frequency relative to unity
        '''

        delays = 2.0 * (dists + self._dist_offset) * np.cos(theta)
        wave_nums = 2.0 * np.pi/ wavelengths
        num_waves = len(wave_nums)
        delays = delays[:,np.newaxis]
        wave_nums = wave_nums[np.newaxis,:]
        config = delays * wave_nums

        # the idea of this method is to take the geometry of the VPM detecotr system and input IQUV
        # and return the band averaged response measured by a detector. 

        det = np.zeros(len(delays))
        ang_dif_factor = 1./2. * np.sin(2. * (alpha - phi))

        stokes = np.array([-1 * Qs * ang_dif_factor * np.sin(2 * phi),
                           -1./4. * Us * (np.sin(2 * (alpha - 2 * phi)) + np.sin(2 * alpha)),
                           Vs * ang_dif_factor])


        #c_config = np.cos(config)
        #s_config = np.sin(config)

        #c_config = np.sum(c_config, axis = 1)/num_waves
        #s_config = np.sum(s_config, axis = 1)/num_waves

        c_config = np.sum(np.cos(config), axis = 1)/ num_waves
        s_config = np.sum(np.sin(config), axis = 1)/ num_waves

        mq = c_config * stokes[0]
        mu = c_config * stokes[1]
        mv = s_config * stokes[2]

        return mq + mu + mv
