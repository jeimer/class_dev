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
        self._tw = 0.1
        self._fc = 1
        return

    def demod(self, param = 'u', fc = 1.):
        s = {'u': self._utrans, 'v': self._vtrans}
        self._tod.data *= s[param][:, self._pos]
        self.lpfilt()
        return

    def set_fc(self, fc):
        self._fc = fc
        return

    def lpfilt(self):
        f_data = np.fft.rfft(self._tod.data)
        order = int(np.ceil(2. / self._tw) * 2)
        t = np.arange(order + 1)
        t = t - order/2
        t[order/2] = 1
        h = np.sin(2 * np.pi * self._fc * t)/t
        h[order/2] = 2 * np.pi * self._fc
        h = h * np.blackman(order + 1)
        h = h / h.sum()
        s = np.zeros( len(self._tod.data))
        s[:len(h)] = h
        imp = np.roll(s, -order/2)
        f_imp = np.fft.rfft(imp)
        self._tod.data = np.fft.irfft(f_data * f_imp, n = len(self._tod.data[0,:]))
        return 
