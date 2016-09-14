import numpy as np
from scipy import optimize
from scipy.signal import butter, lfilter
from moby2.util.mce import MCEButterworth, MCERunfile
from moby2.instruments.class_telescope.products import get_tod
import moby2


def data_valid_edges(tod):
    '''function returns a list of index pairs indicaing the ranges where the TES data from tod.data[0] is nonzero.
    tod: valid moby2 tod object.'''
    array_on_index = np.where(tod.data[0])[0]
    off_edge = np.where((array_on_index[1:]-1) - array_on_index[:-1])[0]
    inner_edges = []
    for element in off_edge:
        inner_edges += [array_on_index[element]]
        inner_edges += [array_on_index[element+1]]
    transitions = np.hstack((array_on_index[0],inner_edges, array_on_index[-1]))
    range_pairs = zip(transitions[0::2], transitions[1::2])
    return range_pairs

def wire_grid_cal_angle(angs):
    '''returns the actual angle of the wire-grid calibrator wires
    angs: (array like) [degrees]'''
    return (angs * 0.9985 - 168.5) % 360

def vpm_direction_ind(vpm_pos):
    '''returns a list of two lists. The first list is the indicies of vpm_pos when the value of vpm_pos is increasing.
    The second list is the indicies of vpm_pos with the vlue is decreasing. In the case when the postion is constant,
    the indicies alternate between the two lists every other time.
    vpm_pos: (list like) list of vpm grid_mirror separations. '''
    dist_inc_ind = []
    dist_dec_ind = []

    for index in range(len(vpm_pos_vector)-1):
        prev_pos = vpm_pos[index]
        new_pos = vpm_pos[index+1]
        increase_score = 0

        if prev_pos < new_pos:
            dist_inc_ind += [index]
        elif prev_pos > new_pos:
            dist_dec_ind += [index]
        elif increase_score % 2 == 0:
            dist_inc_ind += [index]
            increase_score += 1
        else:
            dist_dec_ind += [index]
            increase_score += 1

    return [dist_inc_ind, dist_dec_ind]

def single_pole_lp_filt(freqs, tau):
    '''
    returns the transfer function for a single pole low-pass filter with time constant tau at each frequency in freqs
    freqs: (array like) list of frequencies [Hz]
    tau: (float) or (array like) time constant(s) of filter [seconds]
    '''
    pole = 2.j * np.pi * freqs
    if type(tau) == np.float64 or type(tau) == float:
        return 1./(1 + tau * pole)
    else:
        tau = tau[:,np.newaxis]
        return 1./(1 + tau * pole)

def vpm_direction_ind(vpm_pos):
    '''returns a list of two lists. The first is a list of indicies of vpm_pos_vecter
    where the value is getting larger with index, the second is a list of
    indicies of vpm_pos_vector where the value is getting smaller with index.
    The last point is not returned in either list.
    vpm_pos: (list like)'''
    dist_inc_ind = []
    dist_dec_ind = []
    for index in range(len(vpm_pos)-1):
        prev_pos = vpm_pos[index]
        new_pos = vpm_pos[index+1]
        increase_score = 0
        if prev_pos < new_pos:
            dist_inc_ind += [index]
        elif prev_pos > new_pos:
            dist_dec_ind += [index]
        elif increase_score%2 == 0:
            dist_inc_ind += [index]
            increase_score += 1
        else:
            dist_dec_ind += [index]
            increase_score += 1
    return [dist_inc_ind, dist_dec_ind]

def hyst_metric(y_1, e_1, y_2, e_2):
    '''evaluates the level of hysteresis defined by the sum of the square of the separation in y-values (y_1, y_2),
    deweighted by respective errors (e_1, e_2)
    y_1, y_2: (array like) the y-values of the two branches of the hysteresis loop in question
    e_1, e_2: (array like) the errors on the y-values'''
    val = (y_1 - y_2)**2 / np.sqrt(e_1**2 + e_2**2)
    return val.sum()


def eval_hysteresis2(tau, tod, det_num):
    '''returns the value of hyst_metric for given choice of tau, after being removed from tes_dat. assumeing vpm_dat
    grid mirror separations.
    tau: (float) time constant (seconds)
    tes_dat: (array like) single dimential array of tes time ordered data.
    vpm_dat: (array like) single dimentional array of grid-mirror separation (mm)'''

    vpm_inc, vpm_dec = vpm_direction_ind(tod.vpm)

    f = moby2.tod.filter.TODFilter()
    f.add('deTimeConstant', {'tau': [float(tau)]})
    f.apply(tod, dets = [det_num])

    single_tes = tod.data[det_num] - tod.data[det_num].mean()
    increase_tes = single_tes[vpm_inc]
    decrease_tes = single_tes[vpm_dec]

    #first select bins for entire data set
    hist, bins = np.histogram(tod.vpm,'auto')

    inc_hist, inc_bins = np.histogram(tod.vpm[vpm_inc], bins)
    inc_y, _ = np.histogram(tod.vpm[vpm_inc], bins, weights = increase_tes)
    inc_y2, _ = np.histogram(tod.vpm[vpm_inc], bins, weights = increase_tes * increase_tes)
    dec_hist, dec_bins = np.histogram(tod.vpm[vpm_dec], bins)
    dec_y, _ = np.histogram(tod.vpm[vpm_dec], bins, weights = decrease_tes)
    dec_y2, _ = np.histogram(tod.vpm[vpm_dec], bins, weights = decrease_tes * decrease_tes)

    mid = [(a+b)/2 for a,b in zip(bins[:-1], bins[1:])]
    mean_inc = inc_y / inc_hist
    eom_inc = np.sqrt((inc_y2/inc_hist - mean_inc * mean_inc)/(inc_hist-1))

    mean_dec = dec_y / dec_hist
    eom_dec = np.sqrt((dec_y2/dec_hist - mean_dec * mean_dec)/(dec_hist-1))

    return hyst_metric(mean_inc, eom_inc, mean_dec, eom_dec)



def eval_hysteresis(tau, tes_dat, vpm_dat):
    '''returns the value of hyst_metric for given choice of tau, after being removed from tes_dat. assumeing vpm_dat
    grid mirror separations.
    tau: (float) time constant (seconds)
    tes_dat: (array like) single dimential array of tes time ordered data.
    vpm_dat: (array like) single dimentional array of grid-mirror separation (mm)'''

    vpm_inc, vpm_dec = vpm_direction_ind(vpm_dat)

    print('shape is, ', np.shape(tes_dat))
    defilt_data = remove_tau(tes_dat, tau)

    print('shape is, ', np.shape(defilt_data))

    single_tes = defilt_data - defilt_data.mean()
    increase_tes = single_tes[vpm_inc]
    decrease_tes = single_tes[vpm_dec]

    #first select bins for entire data set
    hist, bins = np.histogram(vpm_dat,'auto')

    inc_hist, inc_bins = np.histogram(vpm_dat[vpm_inc], bins)
    inc_y, _ = np.histogram(vpm_dat[vpm_inc], bins, weights = increase_tes)
    inc_y2, _ = np.histogram(vpm_dat[vpm_inc], bins, weights = increase_tes * increase_tes)
    dec_hist, dec_bins = np.histogram(vpm_dat[vpm_dec], bins)
    dec_y, _ = np.histogram(vpm_dat[vpm_dec], bins, weights = decrease_tes)
    dec_y2, _ = np.histogram(vpm_dat[vpm_dec], bins, weights = decrease_tes * decrease_tes)

    mid = [(a+b)/2 for a,b in zip(bins[:-1], bins[1:])]
    mean_inc = inc_y / inc_hist
    eom_inc = np.sqrt((inc_y2/inc_hist - mean_inc * mean_inc)/(inc_hist-1))

    mean_dec = dec_y / dec_hist
    eom_dec = np.sqrt((dec_y2/dec_hist - mean_dec * mean_dec)/(dec_hist-1))

    return hyst_metric(mean_inc, eom_inc, mean_dec, eom_dec)

def apply_filter(data, filt):
    ''' given configuration domain data and frequency domain filt transfer function, this function returns
    an array of data to which the filter has been applied.
    data: (array like) time domain-like data array
    filt: (array like) transfer function of a filter evaluated at the same frequencies resulting from the fft of the data
    '''
    fft_data = np.fft.fft(data)
    return np.fft.ifft(fft_data * filt).real

def find_tau(tes_dat, vpm_dat):
    '''returns an array of time constant tau values for each detector in tes_dat.
    tes_dat: (array like) tod.data type array.
    vpm_dat: (array like) grid mirror separation (mm)'''
    bound = (0.0009, 0.01)
    if len(np.shape(tes_dat)) == 1:
        res = [optimize.minimize_scalar(eval_hysteresis, bounds = bound, args = (tes_dat, vpm_dat), method = 'Bounded' ).x]
    else:
        res = []
        num_dets = np.shape(tes_dat)[0]
        for det_num in range(num_dets):
            res += [optimize.minimize_scalar(eval_hysteresis, bounds = bound, args = (tes_dat[det_num,:], vpm_dat), method = 'Bounded' ).x]
    return np.array(res)

def find_tau2(tod):
    bound = ((0.0009, 0.01),)
    res = []
    num_dets = np.shape(tod.data)[0]
    for det_num in range(num_dets):
        res1 = optimize.minimize(eval_hysteresis2, [0.004], args = (tod, det_num), bounds = bound)
        res += res1.x

    return np.array(res)

def remove_tau(det_dat, tau):
    '''given a detector time stream, or array of time streams, a single pole filter with time constant tau, or array of taus,
    is deconvolved from the time stream. The recoved time stream(s) is(are) returned.
    det_dat: (array like) first index is detector, second index is time index.
    tau: (array like) time constant of respective detector (seconds)'''

    if len(np.shape(det_dat)) == 1:
        num_dets = 1
        samps = np.shape(det_dat)[0]
    else:
        num_dets = np.shape(det_dat)[0]
        samps = np.shape(det_dat)[1]
    freqs = np.arange(float(samps))/samps
    freqs[int((samps + 1)/2):] -= 1.
    samp_freq = 25e6/100./11./113.
    threshold = 1e-15
    freqs *= samp_freq
    spole = single_pole_lp_filt(freqs, tau)
    #check that spole doesn't have numerically small numbers
    if len(np.where(spole < threshold)[0]) > 0:
        print('filter produces numerically unstable small numbers')
        return 'badness'
    else:
        return apply_filter(det_dat, 1./spole)

def vpm_hyst(tes_data, vpm_data, in_bins):
    ''' returns bin centers, mean values, and error-on-mean values for portion of data when grid-mirror distance is
    increasing and for portion of data when grid-mirror distance is decreasing.
    input:
    tes_data: (array like) single detector time stream
    vpm_data: (array like) respective vpm grid-mirror distances
    in_bins: (list like or keyword) can be end points of bins to be used or np.histogram bin keyword
    returned format:
    [mid, mean_inc, eom_inc, mean_dec, eom_dec] = vpm_hyst(tes_data, vpm_data, in_bins)
    mid = array of centers of bins
    mean_inc = array of mean values for tes_data within respective bin when the grid-mirror distance is increasing
    eom_inc = respective error on mean
    mean_dec = similar to mean_inc but while grid-mirror distance is decreasing
    eom_dec = respective error on mean
    '''
    vpm_inc, vpm_dec = vpm_direction_ind(vpm_data)
    increase_tes = tes_data[vpm_inc]
    decrease_tes = tes_data[vpm_dec]

    hist, bins = np.histogram(vpm_data, in_bins)

    inc_hist, inc_bins = np.histogram(vpm_data[vpm_inc], bins)
    inc_y, _ = np.histogram(vpm_data[vpm_inc], bins, weights = increase_tes)
    inc_y2, _ = np.histogram(vpm_data[vpm_inc], bins, weights = increase_tes * increase_tes)
    dec_hist, dec_bins = np.histogram(vpm_data[vpm_dec], bins)
    dec_y, _ = np.histogram(vpm_data[vpm_dec], bins, weights = decrease_tes)
    dec_y2, _ = np.histogram(vpm_data[vpm_dec], bins, weights = decrease_tes * decrease_tes)

    mid = [(a+b)/2 for a,b in zip(inc_bins[:-1], inc_bins[1:])]
    mean_inc = inc_y / inc_hist
    eom_inc = np.sqrt((inc_y2/inc_hist - mean_inc * mean_inc)/(inc_hist-1))

    mean_dec = dec_y / dec_hist
    eom_dec = np.sqrt((dec_y2/dec_hist - mean_dec * mean_dec)/(dec_hist-1))

    return [mid, mean_inc, eom_inc, mean_dec, eom_dec]

def get_tod_chunk(path, chunk = 0):
    '''
    loads a tod even when a runfile is not pressent for the full tod.
    A tod is returned with the runfile forced to be the first valid runfile in the passed path.
    path: (string like) full path to targeted tod.
    '''
    tod = get_tod(path)
    runfile_tod = tod.get_sync_data('mceq_runfile_id')
    runfile_id = np.unique(runfile_tod)
    exists = runfile_id > 0
    runfile = runfile_id[exists][chunk]
    return get_tod(path, runfile_ctime = runfile)

def debutter_chunk(tod_chunk, runfile):
    '''returns chunk of data with an inverse butterworth filter applied to tod_chunk.
    tod_chunk: (array like) tes data to be debutterworthed. First dimesion is detector, second is time.
    runfile: (string) filename of runfile.

    This function applies the MCEButterworth inverse butterworth filter from the moby2.util.mce library
    using the passed runfile to remove the MCE butterworth filter from the tod_chunk.
    '''
    mce_butter = MCEButterworth.from_runfile(runfile)
    filtered_tod = mce_butter.apply_filter(tod_chunk, decimation = 1./113., inverse = True, gain0 = 1)
    return filtered_tod

def calib_chunk(tod_chunk, ivout, array_data):
    '''
    returns a calibrated chunk of data using the responsivity calculated from the passed ivout.
    tod_chunk: (array like) tes data to be calibrated. First dimension is detector, second is time.
    ivout: (class_sql.IvOut object) ivout object created by moby2.instruments.class_telescope.class_sql.find_iv_out_for_tod function.
    array_data: (library) array_data with keys 'row' and 'col' containing the respective row and col of row of tod_chunk.
    '''
    polarity = -1
    dac_bits = 14
    M_ratio = 24.6
    Rfb = 5100.
    filtgain = 2048.
    rc = zip(array_data['row'], array_data['col'])
    rc = [list(el) for el in rc]
    resp = ivout.resp_fit
    resp_rc = [resp[p[0],p[1]] for p in rc]
    resp_rc = np.array(resp_rc)
    resp_rc = resp_rc[:,np.newaxis]
    dI_dDAC = 1./2.**dac_bits/M_ratio/Rfb/filtgain
    cal_tod_chunk = tod_chunk * dI_dDAC * polarity * resp_rc * 1e3 # nv -> pW
    return cal_tod_chunk

def butter_highpass_filter(data, fc, f_samp, order = 5):
    '''
    applies a highpass butterworth filter with fc cutoff of specified order to data
    Parameters:
    data: (array like) tes data to be filtered
    fc: (float) frequency at which the gain drops to 1/sqrt(2). Units must match f_samp units.
    f_samp: (float) sampling frequency. Units must match fc units.
    order: (int) order of Butterworth filter to apply. Default is 5

    Returns:
    filtered_data: (array like) filtered tes data
    '''
    nyq = 0.5 * f_samp
    low = fc/nyq
    b, a = butter(order, low, btype = 'highpass')
    return lfilter(b, a, data)

def good_det_ids(array_data, bad_row = [], bad_col = []):
    num_dets = len(array_data['det_type'])
    good_dets = []
    for det in range(num_dets):
        if array_data['det_type'][det] == 'H':
            good_dets += [det]
        elif array_data['det_type'][det] == 'V':
            good_dets += [det]
    bad_dets = []
    for det in good_dets:
        if array_data['row'][det] in bad_row:
            bad_dets += [det]
        if array_data['col'][det] in bad_col:
            bad_dets += [det]
    bad_dets = np.unique(bad_dets)
    for det in bad_dets:
        good_dets.remove(det)
    return good_dets
