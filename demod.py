import numpy as np


class Demodulator(object):
    def __init__(self, tod):
        self._tod = tod
        self._dets = tod.det_uid
        self._utrans_path = 'umat.npy'
        self._vtrans_path = 'vmat.npy'
        self._bins_path = 'bins.npy'
        self._utrans = np.load(self._utrans_path)[self._dets,:]
        self._vtrans = np.load(self._vtrans_path)[self._dets,:]
        self._bins = np.load(self._bins_path)
        self._sampling_freq = 25e6/100./11/113.
        self._tw = 0.01 / self._sampling_freq
        return



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
        return np.roll(s, -order/2)

    def hpkern(self, fc, n):
        hp = -self.lpkern(fc, n)
        hp[0] = hp[0] + 1
        return hp

    def b_stopkern(self, fcl, fch, n):
        return self.lpkern(fcl, n) + self.hpkern(fch, n)

    def b_passkern(self, fcl, fch, n):
        bp = -self.b_stopkern(fcl, fch, n)
        bp[0] = bp[0] + 1
        return bp

    def filt(self, kern, data, fc):
        '''
        kern: filter func to apply to data
        data to be filtered. np array.
        fc: cuttoff freq in units of sampling freq
        '''
        f_data = np.fft.rfft(data)
        s = np.zeros(np.shape(data))
        print(np.shape(s))
        h = kern(fc, np.shape(data)[-1])
        s[:,:len(h)] = h
        f_imp = np.fft.rfft(s)
        return np.fft.irfft(f_data * f_imp, np.shape(data)[-1])

    def demod(self, param = 'u', fcl = 1.):
        fc = fcl/self._sampling_freq
        s = {'u': self._utrans, 'v': self._vtrans}
        pos = np.digitize(self._tod.vpm - 0.0001/2, self._bins)
        self._tod.data *= s[param][:, pos]
        self._tod.data = self.filt(self.lpkern, self._tod.data, fc)
        return

    def demod2(self, param = 'u', fh = 7., fl = 1.):
        fh = fh/self._sampling_freq
        fl = fl/self._sampling_freq
        s = {'u': self._utrans, 'v': self._vtrans}
        print('data shape is ', np.shape(self._tod.data))
        self._tod.data = self.filt(self.hpkern, self._tod.data, fh)
        print('vpm shape 1 ', np.shape(self._tod.vpm))
        self._tod.vpm.reshape(1, len(self._tod.vpm))
        print('vpm shape 1 ', np.shape(self._tod.vpm))
        self._tod.vpm = self.filt(self.hpkern, self._tod.vpm, fh)
        pos = np.digitize(self._tod.vpm - 0.0001/2, self._bins)
        self._tod.data *= s[param][:, pos]
        self._tod.data = self.filt(self.lpkern, self._tod.data, fl)
        return


