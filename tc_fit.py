import numpy as np
from scipy import optimize

from moby2.tod import filter
import classtools.better_binner as bb

class Tc():

    def __init__(self, tod, trim = 100):
        '''
        initialize Tc object for fitting time constants in a tod. 
        '''
        self._n = len(tod.vpm)
        self._vpm = np.copy(tod.vpm)
        self._trim = trim
        self._tvpm = self._vpm[trim:-trim]
        self._imask = np.zeros(len(self._tvpm), dtype = bool)
        self._dmask = np.zeros(len(self._tvpm), dtype = bool)
        d = np.diff(self._tvpm)
        self._imask[:-1] = d >= 0
        self._dmask[:-1] = d < 0
        dt = []
        for i in range(len(tod.ctime)-1):
            dt += [tod.ctime[i+1] - tod.ctime[i]]
        self._dt = np.mean(dt)
        self._data = np.copy(tod.data)
        self._tdata = self._data[:,trim:-trim]
        self._hist, self._bins = np.histogram(self._tvpm, 'auto')
        self._mid = [(a + b)/2 for a, b in zip(self._bins[:-1], self._bins[1:])]

    def decon_tau(self, data, tau):
        '''
        deconvolve the time constant tau from the single detector time stream
        Parameters:
         data(np.array): array of detector data to deconvolve
         tau(float): detector time constant in seconds.
        Returns:
         data(np.array): with the time constant deconvolved. 
        '''
        filt = np.ones(self._n, dtype = 'complex128')
        freq = np.fft.fftfreq(self._n, self._dt)
        filt += 1j * 2 * np.pi * tau * freq
        oshape = np.shape(data)
        data = data.reshape(1, len(data))
        filter.apply_simple(data, filt)
        data = data.reshape(oshape)
        return data

    def hyst_metric(self, y1, v1, w1, y2, v2, w2):
        '''
        Evaluates the level of hysteresis defined by the sum of the square of
        the separation in y-values (y_1, y_2), deweighted by respective
        variance (v_1, v_2)
        Parameters:
         y_1, y_2: (array like) the y-values of the two branches
          of the hysteresis loop in question
         e_1, e_2: (array like) the variance  on y-values
         w1, w2 (numpy array): number of points represented by each y1 or y2
          mean respectively. 
        '''
        num = (y1 - y2)**2 * (w1 + w2)
        den = np.sqrt(v1 + v2)
        val = num / den
        return val.sum()/ float(self._n)

    def eval_hyst(self, data, det):
        '''
        Calculates the level of hysteresis of a given detector from a given
        tod.
        Parameters:
         data(array): data for which to evaluate the hysteresis.
         det(int): detector number for which to calculate hysteresis.
        Returns:
         hyst_metric value: (float) The hysteresis metric attempts to quantify
         the amount of hysteresis in the time ordered data by comparing binned
         data when the vpm grid-mirror distance is increasing vs when the
         grid-mirror distance is decreasing.
        '''

        mean_i, var_i, hits_i = bb.binner(self._tvpm[self._imask],
                                          data[self._imask],
                                          bins = self._bins)
        mean_d, var_d, hits_d = bb.binner(self._tvpm[self._dmask],
                                          data[self._dmask],
                                          bins = self._bins)
        return self.hyst_metric(mean_i, var_i, hits_i,
                                mean_d, var_d, hits_d)

    def eval_decon(self, tau, det):
        temp_data = np.copy(self._data[det])
        temp_data = self.decon_tau(temp_data, tau)
        return self.eval_hyst(temp_data[self._trim:-self._trim], det)

    def fit_single_tau(self, det):
        #res1 = optimize.minimize(self.eval_decon, [0.004], args = (det))
        res = optimize.least_squares(self.eval_decon, 0.004,
                                     bounds = (0.001, 0.006),
                                     args = ([det]))
        return res.x, res.cost

    def fit_taus(self):
        taus = []
        res = []
        for det in range(np.shape(self._data)[0]):
            tau, cost = self.fit_single_tau(det)
            taus += [tau]
            res += [cost]
        return taus, res




