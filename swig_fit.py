import numpy as np
import csv
import pickle
import os.path
from scipy import stats
from datetime import datetime

import moby2
from moby2.tod import filter, fft
from moby2.instruments.class_telescope.products import get_tod
from moby2.instruments.class_telescope import vpm, class_filters, calibrate
from moby2.instruments.class_telescope.dpkg_util import DpkgLoc, DpkgSpan


import classtools.better_binner as bb
import tau_fit

class Swig():

    def __init__(self, paths, ang_path, skip_meas = None, det_mask = None):
        '''
        generate swig dictionary with data from paths and corresponding angles
        from angs.
        Parameters:
        paths(list): list of paths to dir files holding swig data
        ang_path(str): path to .csv file holding range of times of specific
        swig orientations. File name format is assumed to be
        YYYYMMDD_swig_elXX_bsXX_on.csv where the XX are the elevation and
        boresight of the respective swig measurement in degrees.

        This class is developed for SWiG measurements made after September 2016.
        Prior measurements had a different format and must be analyzed with
        different methods.
        '''

        #data source
        self._paths = paths
        self._ang_path = ang_path
        self._skip_meas = skip_meas
        self._det_m = det_mask

        #parse path for datetime
        ds = map(int, paths[0].split('/')[-2].split('-'))
        self._yr, self._mo, self._d = ds[0:3]
        self._tau_path = 'tau_' + str(self._yr) +'_'+ str(self._mo) +'_'+ str(self._d) + '.p'

        #parse and calibrate angles recorded from field
        self._ct_pairs, angs = self.load_angs()
        self._angs = self.cal_angle(angs)

        if skip_meas != None:
            self._ct_pairs = np.delete(self._ct_pairs, skip_meas, axis = 0)
            self._angs = np.delete(self._angs, skip_meas, axis = 0)

        #package data into dictionary
        self._m_dict = {key:[] for key in np.unique(self._angs)}
        path_num = 0

        for ang_num in range(len(self._angs)):
            ind = []
            m_paths = []
            temp_tod = get_tod(paths[path_num])
            f_times = temp_tod.ctime
            start, stop = self._ct_pairs[ang_num]
            if start > f_times.max():
                path_num += 1
                temp_tod = get_tod(paths[path_num])
                f_times = temp_tod.ctime
            ind += [np.argwhere(f_times > start).min()]
            m_paths += [paths[path_num]]
            if stop > f_times.max():
                path_num += 1
                temp_tod = get_tod(paths[path_num])
                f_times = temp_tod.ctime
            ind += [np.argwhere(f_times < stop).max()]
            m_paths += [paths[path_num]]
            self._m_dict[self._angs[ang_num]] += [self.repack(m_paths, ind)]

        #make cuts dictionary.
        self._cuts = {key:[] for key in self._m_dict}
        for k in self._m_dict:
            count = 0
            for tod in self._m_dict[k]:
                s = np.shape(tod.data)
                self._cuts[k] += [np.array([True] * s[0])]
                count += 1
        #pre-process data
        self._decon_readout = False
        self._decon_taus = False
        self._calc_taus = False
        self._demod = False
        self._cal = False
        return

    def gen_cuts(self):
        self.length_filt()
        self.tau_filt()
        self.raw_var_filt()
        self.demod_var_filt()
        return

    def get_org_ang(self):
        angs = {k:0 for k in self._m_dict}
        for k in angs:
            if k + 168.5 < 360:
                angs[k] = (k + 168.5) / 0.9985
            else:
                angs[k] = (k - 360 + 168.5) / 0.9985
        return angs

#        return (angs * 0.9985 - 168.5) % 360.

    def length_filt(self, min_len = 6000):
        for k in self._m_dict:
            count = 0
            for tod in self._m_dict[k]:
                b = len(tod.vpm) > min_len
                self._cuts[k][count][:] *= b
                count += 1
        return

    def get_time_edges(self):
        edges = {key:[] for key in self._m_dict}
        for k in self._m_dict:
            for tod in self._m_dict[k]:
                start = datetime.fromtimestamp(tod.ctime[0]).strftime( '%H:%M:%S')
                end = datetime.fromtimestamp(tod.ctime[-1]).strftime( '%H:%M:%S')
                edges[k] += [[start, end]]
        return edges

    def tau_filt(self, tau_d = None):
        if tau_d == None:
            tau_d = self._taus
        t_low = 0.001
        t_hi = 0.01
        median_taus = []
        num_taus = []
        proto_entry = tau_d[tau_d.keys()[0]][0]
        det_taus = {det:[] for det in range(len(proto_entry))}
        for k in tau_d:
            for taus in tau_d[k]:
                for det in range(len(taus)):
                    det_taus[det] += [taus[det]]
        medians = []
        means = []
        var = []
        iqr = []
        for k in det_taus:
            medians += [np.median(det_taus[k])]
            means += [np.mean(det_taus[k])]
            var += [np.var(det_taus[k])]
            iqr += [stats.iqr(det_taus[k])]

        #t_low = np.array(means) - np.array(iqr)
        #t_hi = np.array(means) + np.array(iqr)
        t_low = np.array(means) - 4 * np.sqrt(np.array(var))
        t_hi = np.array(means) + 4 * np.sqrt(np.array(var))
        for k in tau_d:
            count = 0
            for taus in tau_d[k]:
                b = (taus > t_low) * (taus < t_hi)
                self._cuts[k][count] *= b
                count += 1
        return medians, means, var, iqr

    def det_view(self, det):
        times = []
        data = []
        for k in self._m_dict:
            for tod in self._m_dict[k]:
                np.hstack((times, tod.ctimes))
                np.hstack((data, tod.data[det]))
        return times, data

    def cut_dict_bounds(self, val_dict, upper, lower):
        for k in self._m_dict:
            count = 0
            for tod in self._m_dict[k]:
                v = val_dict[k][count]
                b = (v > lower) * (v < upper)
                self._cuts[k][count] *= b
                count += 1
        return

    def var_dict(self, trim = 100):
        var = {k:[] for k in self._m_dict}
        for k in var:
            for tod in self._m_dict[k]:
                var[k] += [np.var(tod.data, axis = 1)]
        return var

    def raw_var_filt(self):
        self.cal_swig()
        v_low = 0.0
        v_hi = 0.1
        var = self.var_dict()
        self.cut_dict_bounds(var, v_hi, v_low)

    # def raw_var_filt(self):
    #     self.cal_swig()
    #     v_low = 0
    #     v_hi = 0.1
    #     var = {key:[] for key in self._m_dict}
    #     for k in self._m_dict:
    #         count = 0
    #         for tod in self._m_dict[k]:
    #             v = np.var(tod.data, axis = 1)
    #             var[k] += [v]
    #             b = (v > v_low) * (v < v_hi)
    #             self._cuts[k][count] *= b
    #             count += 1
    #     return var

    # def demod_var_filt(self):
    #     self.cal_swig()
    #     self.demod_swig()
    #     v_low = 0
    #     v_hi = 1e-4
    #     demod_var = {key:[] for key in self._m_dict}
    #     for k in self._m_dict:
    #         count = 0
    #         for tod in self._m_dict[k]:
    #             v = np.var(tod.data, axis = 1)
    #             demod_var[k] += [v]
    #             b = (v > v_low) * (v < v_hi)
    #             self._cuts[k][count] *= b
    #             count += 1
    #     return demod_var

    def demod_var_filt(self):
        self.cal_swig()
        self.demod_swig()
        v_low = 0
        v_hi = 1e-4
        var = self.var_dict()
        self.cut_dict_bounds(var, v_hi, v_low)

    def cal_swig(self):
        if self._cal:
            return
        else:
            for k in self._m_dict:
                for tod in self._m_dict[k]:
                    cal = calibrate.Calib(tod)
                    cal.calib_dP()
        self._cal = True
        return

    def demod_swig(self):
        if self._demod:
            return
        else:
            for k in self._m_dict:
                for tod in self._m_dict[k]:
                    temp_vpm = vpm.Demodulator(tod, twh = 0.5, twl = 0.5)
                    temp_vpm.demod2()
            self._demod = True
        return

    def load_angs(self):
        '''
        Loads the start and stop times from a SWiG measurment from the CSV path.
        Returns an array of pairs of the c_time values of the respective start
        and stop times.
        Returns:
         ct_paris: (list) list of pairs of starting and stopping times of a
          single angle measurement
         angles: (list) list containing the angle of the sparse grid for the
          respective measurement in degrees.
        '''
        yr, mo, d = self._yr, self._mo, self._d
        f = open(self._ang_path, 'rU')
        csv_f = csv.reader(f)
        line_count = 0
        utc_pairs = []
        angles = []
        for row in csv_f:
            if line_count >= 2:
                start = map(int, row[0].split(':'))
                stop = map(int, row[1].split(':'))
                utc_pairs += [[datetime(yr, mo, d, start[0], start[1], start[2]),
                               datetime(yr, mo, d, stop[0], stop[1], stop[2])]]
                angles += [float(row[2])]
            line_count += 1
        b = datetime(1970, 1, 1, 0,0,0,0)
        ct_pairs = []
        for pair in utc_pairs:
            ct_pairs += [[int((pair[0] - b).total_seconds()),
                          int((pair[1] - b).total_seconds())]]
        f.close()
        return np.array(ct_pairs), np.array(angles)

    def cal_angle(self, angs):
        '''
        Returns the actual angle of the wire-grid calibrator wires
        Parameters:
        angs: (array like) [degrees]
        '''
        return (angs * 0.9985 - 168.5) % 360.

    def repack(self, paths, ind):
        '''
        creates a tod spanning from paths[0][ind[0]] to paths[1][ind[1]]
        Parameters:
         paths(list): list of full paths to the dirfile to be processed
         ind(list): ind[0] is the index of paths[0] from which to begin the new
           tod. ind[1] is the ending index from paths[1].
        Returns:
         tod(moby2 tod): a tod object with data originating from in the specified
        path, but containing only the data between the start and stop indices.
        '''
        start_path = paths[0]
        if len(paths) == 1:
            end_path = start_path
        else:
            end_path = paths[1]
        tod_start = DpkgLoc(start_path, ind[0])
        tod_stop = DpkgLoc(end_path, ind[1])
        span = DpkgSpan(tod_start, tod_stop)
        temp_tod = span.get_tod()
        runfile_ids = np.unique(temp_tod.get_sync_data('mceq_runfile_id'))
        tod = span.get_tod(runfile_ctime = runfile_ids[0],
                           vpm_enc_func = vpm.get_raw_vpm_enc_2,
                           det_uid_mask = self._det_m)
        tod.data = np.require(tod.data, requirements = ['C', 'A'])
        return tod

    def tau_dict(self):
        taus = {key: [] for key in self._m_dict.keys()}
        for key in self._m_dict:
            tod_count = 0
            for tod in self._m_dict[key]:
                t = tau_fit.TauFit(tod)
                t, res, suc = t.fit_taus()
                self._cuts[key][tod_count][:] *= suc
                taus[key] += [t]
                tod_count += 1
        self._taus = taus
        self._calc_taus = True
        return taus

    def b_pass_tau(self, fcl = 2., fch = 20.):
        '''
        experience has shown that band-pass filtering TODs can improve the
        results of fitting for a detector time constant. This function steps
        through each tod in the object dictionary, removes a low-order
        polynomial, band filters the data, fits a time constant, tau, and
        returns these as a dictionary matching the format of the SWiG dictionary.
        '''
        taus = {key: [] for key in self._m_dict.keys()}
        for key in self._m_dict:
            tod_count = 0
            for tod in self._m_dict[key]:
                unfiltered_data = np.copy(tod.data)
                k_len = len(tod.data[-1])
                f = class_filters.FIRImpulse(k_len)
                tw = 0.3
                bp_kern = f.b_passkern(twl = tw, twh = tw, fcl = fcl, fch = fch)
                bp_kern = (bp_kern.astype('float32')).reshape(1, k_len)
                filt = fft(bp_kern)
                filter.apply_simple(tod.data, np.squeeze(filt))
                order = f.get_win_sinc_order(tw)
                t = tau_fit.TauFit(tod, trim = order)
                t, res, suc = t.fit_taus()
                self._cuts[key][tod_count][:] *= np.array(suc, dtype = np.bool)
                taus[key] += [t]
                tod.data = unfiltered_data
                tod_count += 1
        self._taus = taus
        self._calc_taus = True
        return taus

    def decon_readout(self):
        if self._decon_readout:
            return
        else:
            for key in self._m_dict:
                for tod in self._m_dict[key]:
                    moby2.tod.filter.prefilter_tod(tod, detrend = False)
            self._decon_readout = True
        return

    def decon_taus(self, fcl = 2., fch = 20.):
        if self._decon_taus:
            return
        else:
            if os.path.isfile(self._tau_path):
                self._taus = pickle.load(open(self._tau_path, 'rb'))
                self._calc_taus = True
            if not self._calc_taus:
                self.b_pass_tau(fcl, fch)
                pickle.dump(self._taus, open(self._tau_path, 'wb'))
            for key in self._m_dict:
                count = 0
                for tod in self._m_dict[key]:
                    taus = self._taus[key][count]
                    moby2.tod.filter.prefilter_tod(tod, deconvolve_readout = False, time_constants = taus)
                    count += 1
            self._decon_taus = True
        return

    def pre_process_swig(self):
        self.decon_readout()
        self.decon_taus()
        self.cal_swig()
        self.gen_cuts()
        return

    def cuts(self):
        return self._cuts
