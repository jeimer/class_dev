import numpy as np
from moby2.tod import filter

class Tc():

    def __init__(self, tod):
        self._n = len(tod.vpm)
        self._imask = np.zeros(len(tod.vpm), dtype = bool)
        self._dmask = np.zeros(len(tod.vpm), dtype = bool)
        d = np.diff(tod.vpm)
        self._imask[:-1] = d >= 0
        self._dmask[:-1] = d < 0
        dt = []
        for i in range(len(tod.ctime)-1):
            dt += [tod.ctime[i+1] - tod.ctime[i]]
        self._dt = np.mean(dt)

    def decon_tau(self, data, tau):
        filt = np.ones(self._n, dtype = 'complex128')
        freq = np.fft.fftfreq(self._n, self._dt)
        filt += 1j * 2 * np.pi * tau * freq
        oshape = np.shape(data)
        data = data.reshape(1, len(data))
        filter.apply_simple(data, filt)
        data = data.reshape(oshape)
        return data

    def hyst_metric(self, y1, v1, y2, v2):
        '''evaluates the level of hysteresis defined by the sum of the square of
        the separation in y-values (y_1, y_2), deweighted by respective
        variance (e_1, e_2) y_1, y_2: (array like) the y-values of the two branches
        of the hysteresis loop in question e_1, e_2: (array like) the variance  on
        the y-values'''
        num = (y1 - y2)**2
        den = np.sqrt(v1 + v2)
        val = num / den
        return val.sum()

    def eval_hyst(self, taus, det):
        '''Calculates the level of hysteresis of a given detector from a given 
        tod once a specified time-constant has been removed from the data.
        Parameters:
         tau: (float) detector time constant to be deconvolved from the data (sec)
         in_tod: (tod object) moby2 tod object.
        det_num: (int) detector number of the device to be evaluated.
        Returns:
        hyst_metric value: (float) The hysteresis metric attempts to quantify the
        ammount of hysteresis in the time ordered data by comparing binned data
        when the vpm grid-mirror distance is increasing vs when the grid-mirror
        distance is decreasing.
        '''

        imask, dmask = self.vpm_direction_ind(in_tod.vpm)
        tod = self._tod.copy()

        f = moby2.tod.filter.TODFilter()
        f.add('deTimeConstant', {'tau': [float(tau)]})
        f.apply(tod, dets = [det])

        #moby2.tod.filter.prefilter_tod(tod, time_constants = taus)

        #first select bins for entire data set
        hist, bins = np.histogram(tod.vpm,'auto')
        mean_i, var_i, hits_i = bb.binner(tod.vpm[imask], tod.data[det, imask],
                                                bins = bins)
        mean_d, var_d, hits_d = bb.binner(tod.vpm[dmask], tod.data[det, dmask],
                                                bins = bins)
        mid = [(a+b)/2 for a,b in zip(bins[:-1], bins[1:])]

        return self.hyst_metric(mean_i, var_i, mean_d, var_d)
