#!/usr/local/bin/env python

import numpy as np
import si_constants
from scipy import signal
import vpm_lib


class Modulator():
    '''
    The Modulator object is an interface to the VPM object for producing product
    time streams and demodulated time streams.
    '''

    def __init__(self, tod, vpm, angles = None):
        '''
        initiates a Demod object.
        Parameters:
        tod: (moby2 tod object) object containing tods to demodulate
        vpm: (VPM object) model of the VPM transfer functions.
        '''
        self._tod = tod
        self._vpm = vpm

        self._num_waves = 200
        self._weights = np.ones(self._num_waves)
        self._freq_low = 33e9
        self._freq_hi = 42e9
        self._sampling_freq = 25e6/100./11./113.

        self._wavelens = si_constants.SPEED_C/ np.linspace(self._freq_low,
                                                           self._freq_hi,
                                                           self._num_waves)


        if angles == None:
            el_offs = tod.info.array_data['el_off']
            az_offs = tod.info.array_data['az_off']

            #bug in vpm_lib.az_el_to_vpm_angs. for now hard code angles on vpm
            #self._theta, self._phi =vpm_lib.az_el_to_vpm_angs(el_offs, az_offs)
            self._theta = np.ones(len(el_offs)) * 20 * np.pi/180
            self._phi = np.ones(len(el_offs)) * np.pi/2
            self._alpha = tod.info.array_data['rot'] * np.pi/180
        else:
            self._theta, self._phi, self._alpha = angles

        self._calibrated = False

    def windowed_sinc(self, width, cutoff):
        '''
        width: (float) width of transition [Hz]
        cutoff: (float) frequency of transition [Hz]

        has phase delay of half 4./width rounded up to nearest even.
        '''
        M = 4. / width
        M = int(np.ceil( M / 2) * 2)
        print( ' length of filter is :', M)
        fc = cutoff/self._sampling_freq
        samps = np.arange(M)
        samps[M/2] = 1
        h = np.sin(2 * np.pi* fc * (samps - M/2))/(samps - M/2) * \
            (0.42 - 0.5 * np.cos(2 * np.pi * samps/ M) + \
             0.08 * np.cos(4 * np.pi * samps/ M))
        h[M/2] = 2 * np.pi * fc
        tot = h.sum()
        return  h / tot # normalize filter

    def windowed_sinc_dft(self, width, cutoff, samps):
        '''
        width: (float) width of transition [Hz]
        cutoff: (float) frequency of transition [Hz]
        samps: (int) length of time-ordered-data to be filtered

        assumes windowed_sinc with phase delay of half 4./windth rounded up to
        nearest even.
        '''
        filt_imp_resp = self.windowed_sinc(width, cutoff)
        t = np.zeros(samps)
        t[0:len(filt_imp_resp)] = filt_imp_resp
        return np.fft.rfft(t)

    def vpm_trans(self, det_num, i_in, q_in, u_in, v_in):
        '''
        simple wrapper for the vpm transfer function.
        returns the expected mean-subtracted single-detector signal for
        specified incoming stokes vector.
        Parameters:
        det_num: (int) the index of the detector.
        i_in, q_in, u_in, v_in: (float)s the input stokes parameters. This
        function performs no checks on the validity of your input stokes vector.
        Make sure you input something that makes sense.

        Returns:
        mean-subtracted single-detector signal for each vpm position in the
        instaciating tod for the input stokes parameters.
        '''
        dists = self._tod.vpm / 1e3 # because VPM object expects units in [m]
        trans = self._vpm.det_vpm(self._alpha[det_num], self._phi[det_num],
                                  self._theta[det_num], dists, self._wavelens,
                                  self._weights, i_in, q_in, u_in, v_in)
        return trans - trans.mean()

    def u_prod(self, det_num):
        '''
        returns the product of the single-detector tod with 2 * the vpm transfer
        function for pure u input
        '''
        trans = self.vpm_trans(det_num, 1, 0, 1, 0)
        return 2 * self._tod.data[det_num] * trans

    def v_prod(self, det_num):
        '''
        returns the product of the single-detector tod with 2 * the vpm transfer
        function for pure v input
        '''
        trans = self.vpm_trans(det_num, 1, 0, 0, 1)
        return 2 * self._tod.data[det_num] * trans

    def q_prod(self, det_num):
        '''
        returns the product of the single-detector tod with 2 * the vpm transfer
        function for pure q input
        '''
        trans = self.vpm_trans(det_num, 1, 1, 0, 0)
        return 2 * self._tod.data[det_num] * trans
