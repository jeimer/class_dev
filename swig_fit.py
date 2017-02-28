import numpy as np
import csv

from datetime import datetime

import moby2
from moby2.instruments.class_telescope.products import get_tod

from classtools.users.lpp.dpkg_util import DpkgSpan, DpkgLoc
import classtools.better_binner as bb

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

        # assume dir-paths are all on the same day
        # dir file name in format YYYY-MM-DD-HH-MM-SS'
        self._paths = paths
        self._ang_path = ang_path
        self._skip_meas = skip_meas
        self._det_m = det_mask
        ds = map(int, paths[0].split('/')[-2].split('-'))
        self._yr, self._mo, self._d = ds[0:3]

        self._ct_pairs, angs = self.load_angs()
        self._angs = self.cal_angle(angs)
        if skip_meas != None:
            self._ct_pairs = np.delete(ct_pairs, skip_meas, axis = 0)
            self._angs = np.delete(angs, skip_meas, axis = 0)

        self._m_dict = {key:[] for key in np.unique(self._angs)}
        path_num = 0
        m_paths = []
        for ang_num in range(len(self._angs)):
            ind = []
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
        path, but containing only the data between the start and stop indicies.
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
                           det_uid_mask = self._det_m)
        tod.data = np.require(tod.data, requirements = ['C', 'A'])
        return tod
