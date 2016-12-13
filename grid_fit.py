
import emcee
import numpy as np

import tod_ops


def from_data(m_dict, det_num):
    det_data = []
    ang_num = 0
    vpm_pos = []
    measure_vec = []
    for key in m_dict:
        for tod in m_dict[key]:
            det_data += [tod.data[det_num]]
            measure_vec += [ang_num]
            vpm_pos += [tod.vpm]
            ang_num += 1
    return det_data, vpm_pos, measure_vec

def single_visit_form_data(m_dict, det_num, visit_num):
    det_data = []
    vpm_pos = []
    for key in m_dict:
        det_data += [m_dict[key][visit_num].data[det_num]]
        vpm_pos += [m_dict[key][visit_num].vpm]
    return det_data, vpm_pos


def uniform_lnprior(val, min_val, max_val):
    if min_val < val < max_val:
        return 0
    return -np.inf

ln_alpha_prior = lambda alpha: uniform_lnprior(alpha, np.pi/4. * 0.5, np.pi/4. * 1.5)
ln_phi_prior = lambda phi: uniform_lnprior(phi, np.pi/2. * 0.5, np.pi/2. * 1.5)
ln_theta_prior = lambda theta: uniform_lnprior(theta, 5 * np.pi/180., 40 * np.pi/180.)
ln_d_offset_prior = lambda d_offset: uniform_lnprior(d_offset, 0, 0.001)
ln_p_offset_prior = lambda p_offset: uniform_lnprior(p_offset, -0.05, 0.05)
ln_u_prior = lambda u: uniform_lnprior(u, -.15, .15)

def ln_prior(walker, num_angles):
    alpha = walker[0]
    phi = walker[1]
    theta = walker[2]
    d_offset = walker[3]
    p_offsets = walker[4:(4 + num_angles)]
    us = walker[4 + num_angles:]
    if not np.isfinite(ln_alpha_prior(alpha)):
        return -np.inf
    if not np.isfinite(ln_phi_prior(phi)):
        return -np.inf
    if not np.isfinite(ln_theta_prior(theta)):
        return -np.inf
    if not np.isfinite(ln_d_offset_prior(d_offset)):
        return -np.inf
    for case in p_offsets:
        if not np.isfinite(ln_p_offset_prior(case)):
            return -np.inf
    for case in us:
        if not np.isfinite(ln_u_prior(case)):
            return -np.inf
    return 0


def vpm_model(alpha, phi, theta, d_offset, p_offset, u, vpm_pos, wavelengths, weights, vpm):
    vpm_model = []
    for meas_num in range(len(vpm_pos)):
        dist = vpm_pos[meas_num] + d_offset
        vpm_model += [vpm.det_vpm(alpha, phi, theta, dist, wavelengths,
                                  weights, u[meas_num], 0, u[meas_num], 0) + p_offset[meas_num]]
    return vpm_model

def ln_like(walker, vpm_pos, det_data, wavelengths, weights, vpm):
    num_angles = len(vpm_pos)
    alpha = walker[0]
    phi = walker[1]
    theta = walker[2]
    d_offset = walker[3]
    p_offsets = walker[4:(4 + num_angles)]
    us = walker[4 + num_angles:]
    diff = 0
    vpm_m = vpm_model(alpha, phi, theta, d_offset, p_offsets, us, vpm_pos, wavelengths, weights, vpm)
    for setup in range(num_angles):
        diff += np.sum((det_data[setup] - vpm_m[setup])**2)
    return np.sum(diff)

def ln_post(walker, vpm_pos, det_data, wavelengths, weights, vpm):
    lp = ln_prior(walker, len(vpm_pos))
    if not np.isfinite(lp):
        return -np.inf
    return lp + ln_like(walker, vpm_pos, det_data, wavelengths, weights, vpm)


on_paths = ['/data/field/2016-10/2016-10-08/2016-10-08-14-50-00/',
            'data/field/2016-10/2016-10-08/2016-10-08-15-00-00/',
            '/data/field/2016-10/2016-10-08/2016-10-08-15-10-00/']

time_edge_file = '/home/eimer/class/my_dev/calibration_grid_data/20161008_sparse_grid_el45_bs0_on.csv'
tau_file = '/home/eimer/class/my_dev/calibration_grid_data/2016-10-08_BS0_cal_grid.pickle'

num_waves = 200
freq_low = 33e9
freq_hi = 43e9



m_dict, cal_grd_angs = tod_ops.make_sparse_grid_dict2(on_paths, time_edge_file, skip_meas = [0])
tod_ops.pre_filter_sparse_grid_dict(m_dict, tau_file)
s_det_data, s_vpm_pos = single_visit_form_data(m_dict, 0, 0)



wavelengths = si_constants.SPEED_C/np.linspace(freq_low,freq_hi,num_waves)
weights = np.ones(len(wavelengths))
vpm = vpm_ideal.IdealVPM()
vpm_pos = [vpm_data/1e3 for vpm_data in s_vpm_pos]

alpha_0 = np.pi/4.
phi_0 = np.pi/2.
theta_0 = 20 * np.pi/180.
d_offset_0 = 0.00012
p_offset_0 = np.zeros(len(vpm_pos))
u_0 = np.ones(len(vpm_pos)) * 1e-3
p_0 = np.array([alpha_0, phi_0, theta_0, d_offset_0])
p_0 = np.append(p_0, [p_offset_0, u_0])

ndim = 4 + 2 * len(s_vpm_pos)
nwalkers = 2 * ndim + 2
s_alpha = alpha_0/10.
s_phi = phi_0/10.
s_theta = theta_0/10.
s_d_offset = d_offset_0/5.
s_p_offset = 0.001
s_u = 0.1
alpha_w = np.random.normal(alpha_0, s_alpha, (nwalkers,1))
phi_w = np.random.normal(phi_0, s_phi, (nwalkers,1))
theta_w = np.random.normal(theta_0, s_theta, (nwalkers, 1))
d_offset_w = np.random.normal(d_offset_0, s_d_offset, (nwalkers, 1))
p_offset_w = np.random.normal(p_offset_0[0], s_p_offset, (nwalkers, len(s_vpm_pos)))
u_w = np.random.normal(u_0[0], s_u, (nwalkers, len(s_vpm_pos)))
p0 = np.hstack((alpha_w, phi_w, theta_w, d_offset_w, p_offset_w, u_w))

sampler = emcee.EnsembleSampler(nwalkers, ndim, ln_prob, args = (s_vpm_pos, s_det_data), threads = 20)
pos, prob, state = sampler.run_mcmc(p0, 5)





