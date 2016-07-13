import numpy as np
from scipy import optimize

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
    return angs * 0.9985 - 168.5

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
        elif increase_score%2 == 0:
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
    if type(tau) == float:
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

def eval_hysteresis(tau, tes_dat, vpm_dat):
    '''returns the value of hyst_metric for given choice of tau, after being removed from tes_dat. assumeing vpm_dat
    grid mirror separations.
    tau: (float) time constant (seconds)
    tes_dat: (array like) single dimential array of tes time ordered data.
    vpm_dat: (array like) single dimentional array of grid-mirror separation (mm)'''

    vpm_inc, vpm_dec = vpm_direction_ind(vpm_dat)

    defilt = remove_tau(tes_dat, tau)
    #n = tes_dat.shape[-1]
    #freqs = np.arange(float(n))/n
    #freqs[int((n+1)/2):] -= 1.
    #sample_freq = 50e6/200./11.
    #decimation = 1./113.
    #f = freqs * sample_freq * decimation
    #spole = single_pole_lp_filt(f, tau)
    #spole_inv = 1./spole
    #defilt_data = apply_filter(tes_dat, spole_inv)

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
    fft_data = np.fft.fft(data)
    return np.fft.ifft(fft_data * filt).real

def find_tau(tes_dat, vpm_dat):
    '''returns an array of time constant tau values for each detector in tes_dat.
    tes_dat: (array like) tod.data type array.
    vpm_dat: (array like) grid mirror separation (mm)'''
    bound = (0.0009, 0.01)
    print('shape is' + str(np.shape(tes_dat)))
    if np.shape(tes_dat)[0] == 1:
        res = [optimize.minimize_scalar(eval_hysteresis, bounds = bound, args = (tes_dat, vpm_dat), method = 'Bounded' ).x]
    res = []
    num_dets = np.shape(tes_dat)[0]
    for det_num in range(num_dets):
        res += [optimize.minimize_scalar(eval_hysteresis, bounds = bound, args = (tes_dat[det_num,:], vpm_dat), method = 'Bounded' ).x]
    return np.array(res)

def remove_tau(det_dat, tau):
    '''given a detector time stream, or array of time streams, a single pole filter with time constant tau, or array of taus,
    is deconvolved from the time stream. The recoved time stream(s) is(are) returned.
    det_dat: (array like) first index is detector, second index is time index.
    tau: (array like) time constant of respective detector (seconds)'''

    num_dets = np.shape(det_dat)[0]
    samps = np.shape(det_dat)[1]
    freqs = np.arange(float(samps))/n
    freqs[int((n + 1)/2):] -= 1.
    samp_freq = 25e6/100./11./113.
    threshold = 1e-15
    freqs *= samp_freq
    spole = single_pole_pl_filt(freqs, tau)
    #check that spole doesn't have numerically small numbers
    if len(np.where(spole < threshold)[0]) > 0:
        print('filter produces numerically unstable small numbers')
        return 'badness'
    else:
        return apply_filter(det_dat, 1./spole)
