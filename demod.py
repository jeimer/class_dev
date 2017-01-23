import numpy as np


class Demodulator(object):
    def __init__(self, tod):
        self._tod = tod
        self._dets = tod.det_uid
        self._utrans_path = 'umat.npy'
        self._vtrans_path = 'vmat.npy'
        self._bins = 'bins.npy'
        self._utrans = np.load(self._utrans_path)[self._dets,:]
        self._vtrans = np.load(self._vtrans_path)[self._dets,:]
        self._pos = np.digitize(tod.vpm - 0.0001/2, np.load(self._bins))
        self._sampling_freq = 25e6/100./11/113.
        self._tw = 0.01 / self._sampling_freq
        return

    def demod1(self, param = 'u', fc = 1.):
        s = {'u': self._utrans, 'v': self._vtrans}
        self._tod.data *= s[param][:, self._pos]
        fc = fc/self._sampling_freq
        self.lpfilt(self._tod.data, fc)
        return

    def demod2(self, param = 'u', fch = 7, fcl = 1.):
        s = {'u': self._utrans, 'v': self._vtrans}
        #self._tod.data = 
        self._tod.data *= s[param][:, self._pos]
        self.lpfilt()

    def lpkern(self, fc, n):
        order = int(np.ceil(2. / self._tw) * 2)
        t = np.arange(order + 1)
        t = t - order/2
        t[order/2] = 1
        h = np.sin(2 * np.pi * fc * t)/t
        h[order/2] = 2 * np.pi * fc
        h = h * np.blackman(order + 1)
        h = h / h.sum()
        s = np.zeros(n)
        s[:len(h)] = h
        imp = np.roll(s, -order/2)

    def hpkern(self, fc, n):
        hp = -lpkern(fc, n)
        hp[0] = hp[0] + 1
        return hp

    def b_stopkern(self, fcl, fch, n):
        return self.lpkern(fcl, n) + self.hpkern(fch, n)

    def b_passkern(self, fcl, fch, n):
        bp = -self.b_stopkern(fcl, fch, n)
        bp[0] = bp[0] + 1
        return bp

    def lpfilt(self, data, fc):
        '''
        data to be filtered. np array.
        fc: cuttoff freq in units of sampling freq
        '''
        f_data = np.fft.rfft(data)
        s = np.zeros(np.shape(data))
        s[:,:len(h)] = lpkern(fc, np.shape(data)[-1])
        imp = np.roll(s, -order/2, axis = -1)
        f_imp = np.fft.rfft(imp)
        return np.fft.irfft(f_data * f_imp, np.shape(data)[-1])



