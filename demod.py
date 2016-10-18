#!/usr/local/bin/env python

import numpy as np
import si_constants
from scipy import signal
import vpm_lib


class Demod():

    def __init__(self, tod, vpm, angles = None):
        self._tod = tod
        self._vpm = vpm

        self._num_waves = 200
        self._weights = np.ones(self._num_waves)
        self._freq_low = 33e9
        self._freq_hi = 42e9
        self._sampling_freq = 25e6/100./11./113.


        if angles == None:
            el_offs = tod.info.array_data['el_off']
            az_offs = tod.info.array_data['az_off']

            #bug in vpm_lib.az_el_to_vpm_angs. for now hard code angles on vpm
            #self._theta, self._phi = vpm_lib.az_el_to_vpm_angs(el_offs, az_offs)
            self._theta = np.ones(len(el_offs)) * 23 * np.pi/180
            self._phi = np.ones(len(el_offs)) * np.pi/2
            self._alpha = tod.info.array_data['rot'] * np.pi/180
        else:
            self._theta, self._phi, self._alpha = angles

        self._calibrated = False

    def demodulate_ts(self, good_det_indx,  cutoff = 0.5, order = 3):
        wavelens = si_constants.SPEED_C/ np.linspace(self._freq_low, self._freq_hi, self._num_waves)
        nyq = 0.5 * self._sampling_freq
        low = cutoff/ nyq
        b, a = signal.butter(order, low, btype = 'lowpass')

        el_offs = tod.info.array_data['el_off']
        az_offs = tod.info.array_data['az_off']
        alpha = tod.info.array_data['rot']

        if not self._calibrated:
            self.calibrate_vpm()
            self._calibrated = True

        for det_num in good_det_indx:
            tod.data[det_num] = signal.lfilter(b, a, self.u_prod(dists, wavelens, det_num))
        return

    def vpm_trans(self, wavelens, det_num, i_in, q_in, u_in, v_in):
        trans = self._vpm.det_vpm(self._alpha[det_num], self._phi[det_num], self._theta[det_num],
                                  self._tod.vpm / 1e3, wavelens, self._weights, i_in, q_in, u_in, v_in)
        return trans - trans.mean()

    def u_prod(self, wavelens, det_num):
        trans = self.vpm_trans(wavelens, det_num, 1, 0, 1, 0)
        return 2 * self._tod.data[det_num] * trans

    def v_prod(self, wavelens, det_num):
        trans = self.vpm_trans(wavelens, det_num, 1, 0, 0, 1)
        return 2 * self._tod.data[det_num] * trans

    def q_prod(self, wavelens, det_num):
        trans = self.vpm_trans(wavelens, det_num, 1, 1, 0, 0)
        return 2 * self._tod.data[det_num] * trans

    def calibrate_vpm():
        return


