import numpy as np
from moby2.tod import filter

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

    def set_tw(self, tw):
        '''
        set the transition width of the filters
        '''
        self._tw = tw / self._sampling_freq
        return

    def get_tw(self):
        '''
        return transition width of the fiters
        '''
        return self._tw

    def lpkern(self, fc, n):
        '''
        Impulse response of windowed sinc low-pass filter
        Parameters:
        fc(float): cutoff frequency as a fraction of the sampling frequency.
        n(int): required length of the returned filter
        Returns:
        zero-phase windowed-sinc low-pass filter time-domain kernel of
        length n.
        '''
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
        '''
        Impulse response of windowed sinc high-pass filter
        Parameters:
        fc(float): cutoff frequency as a fraction of the sampling frequency.
        n(int): required length of the returned filter
        Returns:
        zero-phase windowed-sinc high-pass filter time-domain kernel of
        length n.
        '''
        hp = -self.lpkern(fc, n)
        hp[0] = hp[0] + 1
        return hp

    def b_stopkern(self, fcl, fch, n):
        '''
        Impulse response of windowed sinc band-stop filter
        Parameters:
        fcl(float): low-edge cutoff frequency as a fraction of the
        sampling frequency.
        fch(float): high-edge cutoff frequency as a fraction of the
        sampling frequency.
        n(int): required length of the returned filter
        Returns:
        zero-phase windowed-sinc band-stop filter time-domain kernel of
        length n.
        '''
        return self.lpkern(fcl, n) + self.hpkern(fch, n)

    def b_passkern(self, fcl, fch, n):
        '''
        Impulse response of windowed sinc band-pass filter
        Parameters:
        fcl(float): low-edge cutoff frequency as a fraction of the
        sampling frequency.
        fch(float): high-edge cutoff frequency as a fraction of the
        sampling frequency.
        n(int): required length of the returned filter
        Returns:
        zero-phase windowed-sinc band-pass filter time-domain kernel of
        length n.
        '''
        bp = -self.b_stopkern(fcl, fch, n)
        bp[0] = bp[0] + 1
        return bp

    def filt(self, kern, data, fc):
        '''
        apply the specified kernal to the data with 
        kern: filter func to apply to data. (currently only works for
        highpass and lowpass kernels)
        data to be filtered. np array.
        fc: cuttoff freq in units of sampling freq
        '''
        f_data = np.fft.rfft(data)
        s = np.zeros(np.shape(data))
        h = kern(fc, np.shape(data)[-1])
        s[:,:len(h)] = h
        f_imp = np.fft.rfft(s)
        return np.fft.irfft(f_data * f_imp, np.shape(data)[-1])

    def demod(self, param = 'u', fcl = 3.):
        '''
        demodulate the instatiated tod.
        *** WARNING ***
        THE TOD IS ALTERED
        Parameters:
        param(string): either 'u' or 'v' to demodulate
        fcl(float): low-pass cuttoff frequency in Hz. The low pass filter is
        applied after taking the product of the tod with the VPM transfer
        function.

        '''
        fc = fcl/self._sampling_freq
        s = {'u': self._utrans, 'v': self._vtrans}
        pos = np.digitize(self._tod.vpm - 0.0001/2, self._bins)
        if np.shape(self._tod.data[0]) == 1:
            self._tod.data = self._tod.data.reshape(1, len(self._tod.data))
        self._tod.data *= s[param][:, pos]
        self._tod.data = self.filt(self.lpkern, self._tod.data, fc)
        return

    def demod2(self, param = 'u', fh = 7., fl = 3.):
        '''
        demodulate the instatiated tod.
        *** WARNING ***
        THE TOD IS ALTERED
        Parameters:
        param(string): either 'u' or 'v' to demodulate
        fh(float): high-pass cuttoff frequency in Hz. The TOD and vpm are both
        high-passed prior to demodulation. 
        fcl(float): low-pass cuttoff frequency in Hz. The low pass filter is
        applied after taking the product of the tod with the VPM transfer
        function.
        '''

        fh = fh/self._sampling_freq
        fl = fl/self._sampling_freq
        s = {'u': self._utrans, 'v': self._vtrans}
        self._tod.data = self.filt(self.hpkern, self._tod.data, fh)
        pos = np.digitize(self._tod.vpm - 0.0001/2, self._bins)
        self._tod.vpm = self._tod.vpm.reshape(1, len(self._tod.vpm))
        self._tod.vpm = self.filt(self.hpkern, self._tod.vpm, fh)
        self._tod.data *= s[param][:, pos]
        self._tod.data = self.filt(self.lpkern, self._tod.data, fl)
        return

    def demod3(self, param = 'u', fh = 7., fl = 3.):
        '''
        demodulate the instatiated tod.
        *** WARNING ***
        THE TOD IS ALTERED
        Parameters:
        param(string): either 'u' or 'v' to demodulate
        fh(float): high-pass cuttoff frequency in Hz. The TOD and vpm are both
        high-passed prior to demodulation. 
        fcl(float): low-pass cuttoff frequency in Hz. The low pass filter is
        applied after taking the product of the tod with the VPM transfer
        function.
        '''

        fh = fh/self._sampling_freq
        fl = fl/self._sampling_freq
        s = {'u': self._utrans, 'v': self._vtrans}
        self._tod.data = self.filt(self.hpkern, self._tod.data, fh)
        pos = np.digitize(self._tod.vpm - 0.0001/2, self._bins)
        #self._tod.vpm = self._tod.vpm.reshape(1, len(self._tod.vpm))
        #self._tod.vpm = self.filt(self.hpkern, self._tod.vpm, fh)
        self._tod.data *= s[param][:, pos]
        self._tod.data = self.filt(self.lpkern, self._tod.data, fl)
        return

    def demod4(self, param = 'u', fh = 7., fl = 3.):
        '''
        demodulate the instatiated tod.
        *** WARNING ***
        THE TOD IS ALTERED
        Parameters:
        param(string): either 'u' or 'v' to demodulate
        fh(float): high-pass cuttoff frequency in Hz. The TOD and vpm are both
        high-passed prior to demodulation. 
        fcl(float): low-pass cuttoff frequency in Hz. The low pass filter is
        applied after taking the product of the tod with the VPM transfer
        function.
        '''

        fh = fh/self._sampling_freq
        fl = fl/self._sampling_freq
        s = {'u': self._utrans, 'v': self._vtrans}
        self._tod.data = self.filt(self.hpkern, self._tod.data, fh)
        oshape = np.shape(self._tod.vpm)
        self._tod.vpm = self._tod.vpm.reshape(1, len(self._tod.vpm))
        self._tod.vpm = self.filt(self.hpkern, self._tod.vpm, fh)
        self._tod.vpm = self._tod.vpm.reshape(oshape)
        pos = np.digitize(self._tod.vpm - 0.0001/2, self._bins)
        self._tod.data *= s[param][:, pos]
        self._tod.data = self.filt(self.lpkern, self._tod.data, fl)
        return

    def demod5(self, param = 'u', fh = 7., fl = 3.):
        '''
        demodulate the instatiated tod.
        *** WARNING ***
        THE TOD IS ALTERED
        Parameters:
        param(string): either 'u' or 'v' to demodulate
        fh(float): high-pass cuttoff frequency in Hz. The TOD and vpm are both
        high-passed prior to demodulation. 
        fcl(float): low-pass cuttoff frequency in Hz. The low pass filter is
        applied after taking the product of the tod with the VPM transfer
        function.
        '''

        fh = fh/self._sampling_freq
        fl = fl/self._sampling_freq
        s = {'u': self._utrans, 'v': self._vtrans}
        #self._tod.data = self.filt(self.hpkern, self._tod.data, fh)
        oshape = np.shape(self._tod.vpm)
        self._tod.vpm = self._tod.vpm.reshape(1, len(self._tod.vpm))
        self._tod.vpm = self.filt(self.hpkern, self._tod.vpm, fh)
        self._tod.vpm = self._tod.vpm.reshape(oshape)
        pos = np.digitize(self._tod.vpm - 0.0001/2, self._bins)
        self._tod.data *= s[param][:, pos]
        self._tod.data = self.filt(self.lpkern, self._tod.data, fl)
        return

    def demod6(self, param = 'u', fh = 7., fl = 3.):
        '''
        demodulate the instatiated tod.
        *** WARNING ***
        THE TOD IS ALTERED
        Parameters:
        param(string): either 'u' or 'v' to demodulate
        fh(float): high-pass cuttoff frequency in Hz. The TOD and vpm are both
        high-passed prior to demodulation. 
        fcl(float): low-pass cuttoff frequency in Hz. The low pass filter is
        applied after taking the product of the tod with the VPM transfer
        function.
        '''

        fh = fh/self._sampling_freq
        fl = fl/self._sampling_freq
        s = {'u': self._utrans, 'v': self._vtrans}



        #self._tod.vpm = self._tod.vpm.reshape(1, len(self._tod.vpm))
        hpfilt = np.fft.fft(self.hpkern(fh, len(self._tod.vpm)))
        #hpfilt = np.fft.fftshift(hpfilt)
        #self._tod.vpm = self._tod.vpm.reshape(1, len(self._tod.vpm))
        filter.apply_simple(self._tod.data, hpfilt)
        filter.apply_simple(self._tod.vpm.astype('float32'), hpfilt)

        pos = np.digitize(self._tod.vpm - 0.0001/2, self._bins)
        self._tod.data *= s[param][:, pos]
        lpfilt = np.fft.fft(self.lpkern(lp, len(self._tod.vpm)))
        #hpfilt = np.fft.fftshift(lpfilt)
        filter.apply_simple(self._tod.data, lpfilt)
        return
