import numpy as np


class Demodulator(object):
    def __init__(self, tod):
        self._tod = tod
        self._dets = tod.det_uid
        self._vpm_utrans_path = 'umat.npy'
        self._vpm_vtrans_path = 'vmat.npy'
        self._bins = 'bins.npy'
        self._utrans = np.load(self._vpm_utrans_path)[dets,:]
        self._vtrans = np.load(self._vpm_vtrans_path)[dets,:]
        self._pos = np.digitize(tod.vpm - 0.0001/2, np.load(self._positions))
        self._tw = 0.01
        self._fc = 1
        return

    def demod(self, param = 'u', fc = 1.):
        s = {'u':[0], 'v':[1], 'uv':[0,1]}
        pos_ind = np.digitize(self._tod.vpm, self._pos)
        if len(s[param] > 1):
            self._tod.data = np.array([[self._tod.data],[self._tod.data]])
            self._tod.data *= np.array([])
        self._tod.data *= self._trans[s[param],:,pos_ind]
        lpfilt()
        return

    def set_fc(self, fc):
        self._fc = fc
        return

    def lpfilt(self):
        f_data = np.fft.rfft(self._tod.data)
        order = int(np.ceil(2./tw) * 2)
        t = np.arange(order + 1)
        t = t - order/2
        t[order/2] = 1
        h = np.sin(2 * np.pi * fc * t)/t
        h[order/2] = 2 * np.pi * fc
        h = h * np.blackman(order + 1)
        h = h / h.sum()
        s = np.zeros( len(self._tod.data))
        s[:len(h)] = h
        imp = np.roll(s, -order/2)
        f_imp = np.fft.rfft(imp)
        self._tod.data = np.fft.irfft(f_data * f_imp, n = len(self._tod.data[0,:]))
        return 
