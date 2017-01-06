import numpy as np
import pickle
import csv
from scipy import optimize
from scipy.signal import butter, lfilter
from datetime import datetime


from moby2.util.mce import MCEButterworth, MCERunfile
from moby2.instruments.class_telescope.products import get_tod, optical_det_mask
from moby2.instruments.class_telescope import calibrate
from moby2.tod import cuts
import moby2
from classtools.users.lpp.dpkg_util import DpkgSpan, DpkgLoc


#####
# Fit tau for SWiG measurement
#####

def vpm_direction_ind(vpm_pos):
    '''returns a tuple of two index masks - an increasing index mask and a
    decreasing index mask respectivly. '''
    inc = np.zeros(len(vpm_pos), dtype = bool)
    dec = np.zeros(len(vpm_pos), dtype = bool)
    d = np.diff(vpm_pos)
    inc[:-1] = d >= 0
    dec[:-1] = d < 0
    return inc, dec

def hyst_metric(y_1, e_1, y_2, e_2):
    '''evaluates the level of hysteresis defined by the sum of the square of
    the separation in y-values (y_1, y_2), deweighted by respective
    errors (e_1, e_2) y_1, y_2: (array like) the y-values of the two branches
    of the hysteresis loop in question e_1, e_2: (array like) the errors on
    the y-values'''
    num = (y_1 - y_2)**2
    den = np.sqrt(e_1**2 + e_2**2)
    val = num / den
    return val.sum()

def eval_hyst(tau, in_tod, det_num):
    '''Calculates the level of hysteresis of a given detector from a givin tod
    once a specified time-constant has been removed from the data.
    Parameters:
    tau: (float) detector time constant to be deconvolved from the data (sec)
    in_tod: (tod object) moby2 tod opject.
    det_num: (int) detector number of the device to be evaluated.
    Returns:
    hyst_metric value: (float) The hysteresis metric attempts to quantify the
    ammount of hysteresis in the time ordered data by comparing binned data
    when the vpm grid-mirror distance is increasing vs when the grid-mirror
    distance is decreasing.
    '''

    imask, dmask = vpm_direction_ind(in_tod.vpm)
    tod = in_tod.copy()

    f = moby2.tod.filter.TODFilter()
    f.add('deTimeConstant', {'tau': [float(tau)]})
    f.apply(tod, dets = [det_num])

    single_tes = tod.data[det_num] - tod.data[det_num].mean()
    increase_tes = single_tes[imask]
    decrease_tes = single_tes[dmask]

    #first select bins for entire data set
    hist, bins = np.histogram(tod.vpm,'auto')
    iv = tod.vpm[imask]
    dv = tod.vpm[dmask]

    inc_hist, inc_bins = np.histogram(iv, bins)
    inc_y, _ = np.histogram(iv, bins, weights = increase_tes)
    inc_y2, _ = np.histogram(iv, bins, weights = increase_tes * increase_tes)
    dec_hist, dec_bins = np.histogram(dv, bins)
    dec_y, _ = np.histogram(dv, bins, weights = decrease_tes)
    dec_y2, _ = np.histogram(dv, bins, weights = decrease_tes * decrease_tes)

    mid = [(a+b)/2 for a,b in zip(bins[:-1], bins[1:])]
    mean_inc = inc_y / inc_hist
    eom_inc = np.sqrt((inc_y2/inc_hist - mean_inc * mean_inc)/(inc_hist-1))

    mean_dec = dec_y / dec_hist
    eom_dec = np.sqrt((dec_y2/dec_hist - mean_dec * mean_dec)/(dec_hist-1))

    return hyst_metric(mean_inc, eom_inc, mean_dec, eom_dec)



def find_tau(tod, bp = None):
    '''returns an array of time constant values for each detector in tod.
    Parameters:
    tod(moby2 tod): tod for which to fit time constants
    pb(None or list): band edges for filter as fractions of sampling frequency
    Returns:
    res(numpy array): The optimization result is an array where each element
    is the best fit time-constant (in seconds) of the respective detector.'''

    #TODO add bandpass filter to TOD prior to fitting.
    if bp != None:
    res = []
    num_dets = np.shape(tod.data)[0]
    for det_num in range(num_dets):
        res1 = optimize.minimize(eval_hyst, [0.004], method = 'Nelder-Mead',
                                 args = (tod, det_num))
        res += [float(res1.x)]
    return np.array(res)

#####
# Repackage SWiG measurment data
#####


def wire_grid_cal_angle(angs):
    '''returns the actual angle of the wire-grid calibrator wires
    angs: (array like) [degrees]'''
    return (angs * 0.9985 - 168.5) % 360.


def get_tod_chunk(path, chunk = 0):
    '''
    loads a tod even when a runfile is not pressent for the full tod.
    A tod is returned with the runfile forced to be the first valid runfile in
    the passed path.
    path: (string like) full path to targeted tod.
    '''
    tod = get_tod(path)
    runfile_tod = tod.get_sync_data('mceq_runfile_id')
    runfile_id = np.unique(runfile_tod)
    exists = runfile_id > 0
    runfile = runfile_id[exists][chunk]
    return get_tod(path, runfile_ctime = runfile)

def repack_chunk(paths, start, stop, det_mask = None):
    '''
    Repackages the portion of a tod between start/stop indicies to be a tod.
    Parameters:
    path(list): full path to the dir file to be processed
    start(int): index of the loaded file that will be the start of the new tod
    stop(int): index of the loaded file that will be the end of the new tod
    det_mask(array): bool array of detectors to load
    Returns:
    tod(moby2 tod): a tod object with data originating from in the specified
    path, but containing only the data between the start and stop indicies.
    '''
    start_path = paths[0]
    if len(paths) == 1:
        end_path = start_path
    else:
        end_path = paths[1]
    tod_start = DpkgLoc(start_path, start)
    tod_stop = DpkgLoc(end_path, stop)
    span = DpkgSpan(tod_start, tod_stop)
    temp_tod = span.get_tod()
    runfile_ids = np.unique(temp_tod.get_sync_data('mceq_runfile_id'))
    tod = span.get_tod(runfile_ctime = runfile_ids[0], det_uid_mask = det_mask)
    tod.data = np.require(tod.data, requirements = ['C', 'A'])
    return tod

def make_swig_dict1(paths, angles, min_chunk_size = 1000):
    '''
    Creates a dictionary holding moby2 tods with calibration grid angle as keys
    for SWiG measurement on 2016-06-29.
    Parameters:
    paths(list): list of full paths to dir files holding the calibration data.
    The list must be in the same order as the anlges parameter.
    angles(list): List of angles at which the calibration grid was measured.
    These angles must be in the order of the paths containing the data. If a
    dir file contains more than one calibration grid angle, the angles meausred
    first must apear first in the angles parameter list.
    min_chunk_size: (int) Minimum size for a valid calibration chunk. In some
    relavent dirfiles, data was taken only for a short time. These brief chunks
    of data are not actual calibration grid measurements and are skipped.
    Returns:
    chunks(dictionary): A dictionary with keys that are the anlges from the
    input angles list. The dictionary elements are lists of tods contining data
    collected when the calibration grid was at the respective angle.
    '''
    chunks = {key: [] for key in angles}
    # find edges when MCE was taking data
    angle_num = 0
    for path in paths:
        temp_tod1 = get_tod(path)
        edges = data_valid_edges(temp_tod1)

        # for each pair of edges, form new tod and load the relavent runfile
        for pair in edges:
            if pair[1] - pair[0] >  min_chunk_size:
                chunks[angles[angle_num]] += [repack_chunk([path], pair[0],
                                                           pair[1])]
                angle_num += 1
    return chunks

def load_swig_csv(year, month, day, path):
    '''
    Loads the start and stop times from a SWiG measurment from the CSV path.
    Returns an array of pairs of the c_time values of the respective start and
    stop time.
    Parameters:
    year: (int) year of the measurement
    month: (int) month of the measurment
    day: (int) day of the measurment
    path: (string) full path to the csv file holding the start and stop times
    of the sparse grid measurment.
    Returns:
    ct_paris: (list) list of pairs of starting and stopping times of a single
    angle measurment
    angles: (list) list containing the angle of the sparse grid for the
    respective measurment.
    '''
    f = open(path, 'rU')
    csv_f = csv.reader(f)
    csv_rows = [row for row in csv_f]
    tcsv_rows = csv_rows[2:]
    csv_rows = [row[0:2] for row in tcsv_rows]
    angles = [float(row[2]) for row in tcsv_rows]
    utc_pairs = []
    for row in csv_rows:
            start = [int(item) for item in row[0].split(':')]
            stop = [int(item) for item in row[1].split(':')]
            utc_pairs += [[datetime(year, month, day, start[0], start[1], start[2]),
                           datetime(year, month, day, stop[0], stop[1], stop[2])]]
    ct_pairs = [ ]
    for pair in utc_pairs:
        ct_pairs += [[int((pair[0] - datetime(1970, 1, 1, 0, 0, 0, 0)).total_seconds()),
                       int((pair[1] - datetime(1970, 1, 1, 0, 0, 0, 0)).total_seconds())]]
    return ct_pairs, np.array(angles)


def make_swig_dict2(dir_paths, ang_path, skip_meas = None,
                           det_mask = None):
    '''
    This function is designed to work with the sparse grid measurment performed
    in Sept 2016 or afterward. Creates a dictionary holding lists of moby2 tods
    with calibration grid angle as keys.
    Parameters:
    dir_paths: (list) list of full paths to dir files holding the detector data
    for the measurment. The list must be in the same order as the anlges
    parameter.
    ang_path: (string) full path to the csv file holding the start time, end
    time, and angles of the sparse grid measurment.
    Returns:
    m_dict: (dictionary) A dictionary with keys that are the anlges from the
    input angles list. The dictionary elements are lists of tods contining data
    collected when the calibration grid was at the respective angle.
    '''
    #assume dir-paths are all on the same day
    date_string = map(int, dir_paths[0].split('/')[4].split('-'))
    #date_string = [int(item) for item in date_string]
    tct_pairs, angles = load_swig_csv(date_string[0], date_string[1],
                                      date_string[2], ang_path)
    #remove the measurments indicated by skip_meas
    ct_pairs = []
    if skip_meas != None:
        for index in range(len(tct_pairs)):
            if index not in skip_meas:
                ct_pairs += [tct_pairs[index]]
        angles = np.delete(angles, skip_meas, axis = 0)
    else:
        ct_pairs = tct_pairs

    angles = wire_grid_cal_angle(angles)
    m_dict = {key:[] for key in np.unique(angles)}
    path_num = 0
    for ang_num in range(len(angles)):
        temp_tod = get_tod(dir_paths[path_num])
        f_times = temp_tod.ctime
        start, stop = ct_pairs[ang_num]
        if start > f_times.max():
            path_num += 1
        else:
            start_index = np.argwhere(f_times > start).min()
            start_path = dir_paths[path_num]
            if stop > f_times.max():
                #assumes the stop point is in the next file in the list
                path_num += 1 
            temp_tod = get_tod(dir_paths[path_num])
            f_times = temp_tod.ctime
            stop_index = np.argwhere(f_times < stop).max()
            stop_path = dir_paths[path_num]
            m_dict[angles[ang_num]] += [repack_chunk([start_path, stop_path],
                                                     start_index, stop_index,
                                                     det_mask)]
    return m_dict, angles

def make_tau_dic(cal_grid_dic):
    '''
    uses find_tau to find the best fit time detector time constant to minimize
    tod hysteresis. For corect interpretation of the time constant, the MCE
    buttorworth filter should be removed before running make_tau_dic.
    Parameters:
    cal_grid_dic(dictionary): excpects a dictionary in the format created by
    make_swig_dict2
    Returns:
    taus(dictionary): with the same keys as the input dictionary. The elements
    are lists containing the best fit time constants for each detector for
    each tod.
    '''
    taus = {key: [] for key in cal_grid_dic.keys()}
    samp_num = 0
    for key in cal_grid_dic:
        for visit in cal_grid_dic[key]:
            print('working on measurement:', samp_num)
            taus[key] += [find_tau(visit)]
            samp_num += 1
    return taus

def prefilter_swig_dict(d_dict, tau_path = None):
    '''
    Deconvolves readout and time constants from the input data dictionary.
    A dictionary containing the respective time constants is returned.
    Parameters:
    d_dict(dictionary): expects dictionary of the format created by
    make_swig_dict2.
    tau_path(string): path to pickled dictionary of time constants as
    generated by make_tau_dic
    Returns:
    taus(dictionary): with the same keys as the input dictionary. the elements
    are lists containing the best fit time constants for each detector for
    each tod. 
    '''
    #deconvolve readout
    for k in d_dict:
        for visit in d_dict[k]:
            visit.cuts = cuts.get_constant_det_cuts(visit)
            moby2.tod.filter.prefilter_tod(visit)
    if tau_path == None:
        taus = make_tau_dic(d_dict)
    else:
        #load detector time constants
        with open(tau_path, 'rb') as handle:
            taus = pickle.load(handle)
    for k in d_dict:
        vis_num = 0
        for visit in d_dict[k]:
            moby2.tod.filter.prefilter_tod(visit,
                                           deconvolve_readout = False,
                                           time_constants = taus[k][vis_num],
                                           detrend = True)
            vis_num += 1
            for det in range(len(visit.data)):
                visit.data[det] = visit.data[det] - visit.data[det].mean()
    for key in d_dict:
        for tod in d_dict[key]:
            cal = calibrate.Calib(tod)
            cal.calib_dP()
    return taus

#####
# Begin filtering tools
#####

def low_pass_win_sinc(tw, fc, n):
    '''
    returns the impulse response of a blackman-windowed sinc low-pass filter
    Paramters:
    tw(float): approximate transition width of the filter roll off
    fc(float): cutoff filter frequency as a fraction of the sampling frequency
    n(int): total length of the returned filter
    Returns:
    s(array): zero-phase version of impulse response of the filter.
    '''
    order = int(np.ceil( 2./ tw) * 2)
    t = np.arange(order + 1)
    t = t - order/2
    t[m/2] = 1
    h = np.sin(2 * np.pi * fc * t)/ t
    h[m/2] = 2 * np.pi * fc
    h = h * np.blackman(order + 1)
    h = h / h.sum()
    s = np.zeros(n)
    s[:len(h)] = h
    return np.roll(s, -order/2)

def high_pass_win_sinc(tw, fc, n):
    '''
    returns the impulse response of a blackman-windowed-sinc high-pass filter
    Paramters:
    tw(float): approximate transition width of the filter roll off
    fc(float): cutoff filter frequency as a fraction of the sampling frequency
    n(int): total length of the returned filter
    Returns:
    hp(array): zero-phase version of impulse response of the filter.
    '''
    hp = -low_pass_win_sinc(tw, fc, n)
    hp[0] = hp[0] + 1
    return hp

def band_stop_win_sinc(tw, flow, fhi, n):
    '''
    returns the impulse response of a blackman-windowed-sinc band-stop filter
    Paramters:
    tw(float): approximate transition width of the filter roll off
    flow(float): lower edge cutoff frequency as a fraction of the sampling
    frequency
    fhi(float): upper edge cutoff frequency as a freaction of the sampling
    frequency
    n(int): total length of the returned filter
    Returns:
    s(array): zero-phase version of impulse response of the filter.
    '''
    return low_pass_win_sinc(tw, flow, n) + high_pass_win_sinc(tw, fhi, n)

def band_pass_wind_sinc(tw, flow, fhi, n):
    '''
    returns the impulse response of a blackman-windowed-sinc band-pass filter
    Paramters:
    tw(float): approximate transition width of the filter roll off
    flow(float): lower edge cutoff frequency as a fraction of the sampling
    frequency
    fhi(float): upper edge cutoff frequency as a freaction of the sampling
    frequency
    n(int): total length of the returned filter
    Returns:
    bp(array): zero-phase version of impulse response of the filter.
    '''
    bp = -band_stop_win_sinc(tw, flow, fhi, n)
    bp[0] = bp[0] + 1
    return bp

def apply_filter(data, impulse):
    '''
    perform fft convolution of an impulse on a data set.
    Parameters:
    data(array): array of the data to be filtered
    impulse(array): impulse response of the filter to convolve with the data.
    The impulse array must be the same length as the data array
    Retuns:
    The data convolved with the filter. 
    '''
    f_data = np.fft.rfft(data)
    f_imp = np.fft.rfft(impulse)
    return np.fft.irfft(f_data * f_imp)
