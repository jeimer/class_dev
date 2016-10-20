#!/usr/local/bin/env python

import numpy as np
import si_constants
from scipy import signal
import vpm_lib


class Demod():
    '''
    The Demod object is an interface to the VPM object for producing product time streams and demodulated
    time streams.
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

        self._wavelens = si_constants.SPEED_C/ np.linspace(self._freq_low, self._freq_hi, self._num_waves)


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

    def demod_u(self, good_det_indx,  cutoff = 0.5, order = 3):
        '''
        rewrites the identified good detector data with a demodulated u time stream.
        Parameters:
        good_det_indx: (list) list of detector numbers to demodulate
        cutoff: (float) butterworth filter cutoff frequency [Hz]
        order: (int) order of the lowpass butterworth filter

        ***WARNING***
        this function overwrites the tod detector data. If you want to preserve tod.data,
        use u_prod(det_num) followed by your own low-pass filter
        '''

        b, a = self.lp_butter(cutoff, order)
        for det_num in good_det_indx:
            tod.data[det_num] = signal.lfilter(b, a, self.u_prod(self._tod.vpm, self._wavelens, det_num))
        return

    def demod_q(self, good_det_indx,  cutoff = 0.5, order = 3):
        '''
        rewrites the identified good detector data with a demodulated q time stream.
        Parameters:
        good_det_indx: (list) list of detector numbers to demodulate
        cutoff: (float) butterworth filter cutoff frequency [Hz]
        order: (int) order of the lowpass butterworth filter

        ***WARNING***
        this function overwrites the tod detector data. If you want to preserve tod.data,
        use q_prod(det_num) followed by your own low-pass filter
        '''
        b, a = self.lp_butter(cutoff, order)
        for det_num in good_det_indx:
            tod.data[det_num] = signal.lfilter(b, a, self.q_prod(self._tod.vpm, self._wavelens, det_num))
        return

    def demod_v(self, good_det_indx,  cutoff = 0.5, order = 3):
        '''
        rewrites the identified good detector data with a demodulated v time stream.
        Parameters:
        good_det_indx: (list) list of detector numbers to demodulate
        cutoff: (float) butterworth filter cutoff frequency [Hz]
        order: (int) order of the lowpass butterworth filter

        ***WARNING***
        this function overwrites the tod detector data. If you want to preserve tod.data,
        use v_prod(det_num) followed by your own low-pass filter
        '''
        b, a = self.lp_butter(cutoff, order)
        for det_num in good_det_indx:
            tod.data[det_num] = signal.lfilter(b, a, self.v_prod(self._tod.vpm, self._wavelens, det_num))
        return

    def demod(self, good_det_indx, cutoff = 0.5, order =3):
        '''
        Returns the demodulated q, u, and v time streams for the  identified good detectors.
        Parameters:
        good_det_indx: (list) list of detector numbers to demodulate
        cutoff: (float) butterworth filter cutoff frequency [Hz]
        order: (int) order of the lowpass butterworth filter
        Returns:
        [q,u,v]: (list) each element of this list is another list. The q list, for example, is a list
        of demodulated data in the same order as the input list of good_det_indx. 
        '''
        b, a =  self.lp_butter(cutter, order)
        q = []
        u = []
        v = []
        for det_num in good_det_indx:
            q += [signal.lfilter(b, a, self.q_prod(self._tod.vpm, self._wavelens, det_num))]
            u += [signal.lfilter(b, a, self.u_prod(self._tod.vpm, self._wavelens, det_num))]
            v += [signal.lfilter(b, a, self.v_prod(self._tod.vpm, self._wavelens, det_num))]
        return [q,u,v]


    def lp_butter(self, cutoff, order):
        '''
        simple wrapper for scipy butterworth filter for the CLASS sampling frequency.
        Parameters:
        cutoff: (float) butterworth filter cutoff frequency [Hz]
        order: (int) order of the lowpass butterworth filter
        Returns:
        b, a: (array, array) numerator, denominator polynomials of the IIR filter.
        '''
        nyq = 0.5 * self._sampling_freq
        low = cutoff / nyq
        b, a = signal.butter(order, low, btype = 'lowpass')
        return b, a

    def vpm_trans(self, det_num, i_in, q_in, u_in, v_in):
        '''
        simple wrapper for the vpm transfer function.
        returns the expected mean-subtracted single-detector signal for specified incoming
        stokes vector.
        Parameters:
        det_num: (int) the index of the detector.
        i_in, q_in, u_in, v_in: (float)s the input stokes parameters. This function performs no
        checks on the validity of your input stokes vector. Make sure you input something that
        makes sense.

        Returns:
        mean-subtracted single-detector signal for each vpm position in the instaciating tod for
        the input stokes parameters.
        '''
        dists = self._tod.vpm / 1e3 # because VPM object expects units in [m]
        trans = self._vpm.det_vpm(self._alpha[det_num], self._phi[det_num], self._theta[det_num],
                                  dists, self._wavelens, self._weights, i_in, q_in, u_in, v_in)
        return trans - trans.mean()

    def u_prod(self, det_num):
        '''
        returns the product of the single-detector tod with 2 * the vpm transfer function
        for pure u input
        '''
        trans = self.vpm_trans(det_num, 1, 0, 1, 0)
        return 2 * self._tod.data[det_num] * trans

    def v_prod(self, det_num):
        '''
        returns the product of the single-detector tod with 2 * the vpm transfer function
        for pure v input
        '''
        trans = self.vpm_trans(det_num, 1, 0, 0, 1)
        return 2 * self._tod.data[det_num] * trans

    def q_prod(self, det_num):
        '''
        returns the product of the single-detector tod with 2 * the vpm transfer function
        for pure q input
        '''
        trans = self.vpm_trans(det_num, 1, 1, 0, 0)
        return 2 * self._tod.data[det_num] * trans

    def calibrate_vpm():
        '''
        "We are all interested in the future, for that is where you and I are going to spend the rest of our
        lives." --The Amazing Criswell
        '''
        return


