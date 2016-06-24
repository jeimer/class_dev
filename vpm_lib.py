#!/usr/local/bin/env python

import si_constants
import numpy as np
import scipy.interpolate as interp

def dist_modulate_sin(times, d_min, d_pp, freq):
    '''returns an array of grid mirror distances for each element of input times np.array.
    Modulation is sinusoidal with d_min minimum distance and d_pp the peak-to-peak amplitude
    of the modulation at freq frequency.
    times: np.array of floats [sec]
    d_min: float [mm]
    d_pp: float [mm]
    freq: float [Hz]
    '''
    return d_min + 0.5 * d_pp * (1 - np.cos(2 * np.pi * freq * times))

def jones_to_mueller(jones):
    '''returns the equivalent mueller matrix to an input jones matrix'''
    stokes_conversion_matrix = np.array([[1.,0,0,1.],[1.,0,0,-1.],[0,1.,1.,0],[0,1j,-1.*1j,0]])
    stokes_conversion_inverse = np.array([[1./2., 1./2.,0,0],[0,0,1./2.,-1.*1j/2.],
                                              [0,0,1./2.,1j/2.],[1./2.,-1./2.,0,0]])
    mueller = np.real(np.dot(stokes_conversion_matrix, np.dot(np.kron(jones, np.conj(jones)),\
                                                          stokes_conversion_inverse)))
    return mueller

def az_el_to_vpm_angs(az,el):
    '''Returns the tuple (theta, phi) in radians where theta is the angle of incidence on the VPM, and
    phi is the angle between the plane of incidence and the plane orthogonal to the VPM and bisecting
    the grid wires.
    az: (radians) the azmuthal offset from the boresight of a pointing direction (assuming the boresight
     is directed toward the horizon.
    el: (radians) the elevation offset from the boresight of the pointing direction (assuming the boresight
     is directed toward the horizon.
    coordinates assume El-over-Az coordiante orientation.
    '''
    c_el = np.cos(el)
    s_el = np.sin(el)
    t_el = np.tan(el)
    c_az = np.cos(az)
    s_az = np.sin(az)
    c_9 = np.cos(np.pi/9.)
    s_9 = np.sin(np.pi/9.)
    denom = c_az * np.sqrt(1 + t_el**2 / c_az**2)

    theta = np.arccos(c_el * c_9 * s_az - s_9 * np.sqrt(1 - c_el**2 * s_az**2) * t_el/denom)
    phi = np.arccos(c_el * s_9 * s_az + c_9 * np.sqrt(1 - c_el**2 * s_az**2) * t_el/denom)

    return (theta,phi)

class VPM(object):
    ''' The VPM object contains the gemetric state of a VPM system and includes
    functionality to calculate the transfer function of the VPM-detector system.

    The object data are parameters of the VPM only.
    The only dynamic variable is the distance between the grid-wires and the mirror.
    All other grid parameters are static, but can be accessed/set using get/set methods.

    The ideal grid assumes the VPM consists of an ideal polarizer backed by an ideal mirror.

    If the VPM is not ideal, the caculated transfer fuction assumes the grid wires are copper plated
    tungsten wires using a thin nickel flash, and assumes the mirror is aluminum.

    '''

    def __init__(self):

        #state parameter
        self._dist = 0 #[m] distance between the wires and the mirror

        #physical parameters
        self._pitch = 160e-6 # [m] distannce between adjacent grid wires
        self._r_wire = 25.4e-6 # [m] radius of tungsten wire
        self._thick = {'w':25.4e-6, 'ni':0.254e-6, 'cu':3e-6}
        self._sigma = {'al':3.65e7, 'cu':5.95e7, 'w':1.88e7, 'ni':1.42e7} # [S/m] from Exper. Tech. in Low-Temp Phys by: White and Meeson
        self._ideal = True
        self._radius = sum(self._thick.itervalues())

    def set_dist(self, dist):
        '''set the grid mirror separtation distance [meters]'''
        self._dist = dist

    def get_dist(self):
        '''return the distance between the VPM grid wires and the mirror'''
        return self._dist

    def get_radius(self):
        '''returns the full wire radius in [m]'''
        return self._radius

    def set_thickness(self, material, thick):
        '''
        set the radial thickness of wire material.
        material: string, periodic tale abreviation of material being set
        thick: float [m], radial thickness of layer
        '''
        self._thick[material.lower()] = thick
        self._radius = sum(self._thick.itervalues())

    def get_thickness(self, material):
        '''
        returns the radial thickness of wire material.
        material: string, periodic table abriviation of layer being queried
        '''
        return self._thick[material.lower()]

    def set_wire_pitch(self, pitch):
        '''set the distance between wires in the VPM grid'''
        self._pitch = pitch

    def get_wire_pitch(self):
        '''return the distance between wires in the VPM grid'''
        return self._pitch

    def set_sigma(self, material, sigma):
        '''set ths conductivity of the material.
        material: string, periodic table abrivation of layer being set.
        sigma: float [S/m], SI conductivity of material. 
        '''
        self._sigma[material.lower()] = sigma

    def get_sigma(self, material):
        '''
        returns the conductivity of the specified material
        material: string, periodic table abriviation of layer being queried.
        '''
        return self._sigma[material.lower()]

    def gamma_te_par(self,theta, wavelengths):
        '''
        theta: float [radians], angle of incidence.
        wavelengths: array-like of floats [m], the wav
        calulates the refelction cooefficient for the TE portion of an incoming field that is parallel to the
        grid wires.'''
        if weights == None:
            weights = np.ones(len(wavelengths))
        gam = np.zeros(len(wavelengths))
        if self._ideal:
            gam = -1.0 * np.ones(len(wavelengths))
        else:
            gam = -1.0 * np.ones(len(wavelengths))
        return gam

    def gamma_te_perp(self,theta,wavelengths):
        '''
        theta: float [radians], angle of incidence.
        wavelengths: array-like of floats [m], the wav
        calulates the refelction cooefficient for the TE portion of an incoming field that is perpindicular to the
        grid wires.'''
        if weights == None:
            weights = np.ones(len(wavelengths))
        gam = np.zeros(len(wavelengths))
        if self._ideal:
            gam = -1.0 * np.exp(4 * np.pi * self._dist * np.cos(theta)/wavelengths)
        else:
            gam = -1.0 * np.exp(4 * np.pi * self._dist * np.cos(theta)/wavelengths)
        return gam

    def gamma_tm_par(self, theta, wavelengths):
        '''
        theta: float [radians], angle of incidence.
        wavelengths: array-like of floats [m], the wav
        calulates the refelction cooefficient for the TM portion of an incoming field that is parallel to the
        grid wires.'''
        if weights == None:
            weights = np.ones(len(wavelengths))
        gam = np.zeros(len(wavelengths))
        if self._ideal:
            gam = -1.0 * np.ones(len(wavelengths))
        else:
            gam = -1.0 * np.ones(len(wavelengths))
        return gam

    def gamma_tm_perp(self, theta, wavelengths):
        '''
        theta: float [radians], angle of incidence.
        wavelengths: array-like of floats [m], the wav
        calulates the refelction cooefficient for the TM portion of an incoming field that is perpindicular to the
        grid wires.'''
        if weights == None:
            weights = np.ones(len(wavelengths))
        gam = np.zeros(len(wavelengths))
        if self._ideal:
            gam = -1.0 * np.exp(4 * np.pi * self._dist * np.cos(theta)/wavelengths)
        else:
            gam = -1.0 * np.exp(4 * np.pi * self._dist * np.cos(theta)/wavelengths)
        return gam

    def det_vpm(self, alpha, phi, theta, dists, wavelengths,Is,Qs,Us,Vs, ideal = True):
        ''' follows the spirit and coordinate conventions of Chuss et al 2012.
        alpha (float):[radians] angle of the detectors w.r.t projected grid wires
        phi (float): [radians] angle of grid wires w.r.t. plane of incidence [radians]
        theta (float): [radians] angle of incidence [radians]
        dist (array like): grid mirror separation [m]
        wavelength (array like): [m]
        '''
        self._ideal = ideal

        delays = 2.0 * dists * np.cos(theta)
        wave_nums = 2.0 * np.pi/ wavelengths

        # the idea of this method is to take the geometry of the VPM detecotr system and input IQUV
        # and return the band averaged response measured by a detector. 

        det = np.zeros(len(delays))
        if self._ideal == True:
            mi = 1./2 * Is
            mq1 = Qs * (-1./4 * np.cos(2 * alpha) - 1./4 * np.cos(2*(alpha - 2*phi)))
            ang_dif_factor = 1./2 * np.sin(2 * (alpha - phi))
            for delay_count in range(len(delays)):
                mq2 = Qs * (-1. * ang_dif_factor * np.cos(delays[delay_count] * wave_nums) * np.sin(2 * phi))
                mu1 = Us * (-1./2 * np.sin(2*(alpha - 2*phi)) * np.cos(delays[delay_count] * wave_nums/2.)**2 +
                            1./2 * np.sin(2 * alpha) * np.sin(delays[delay_count] * wave_nums/2.)**2)
                mv1 = Vs * ang_dif_factor * np.sin(delays[delay_count] * wave_nums)
                det_val = mi + mq1 + mq2 + mu1 + mv1
                det[delay_count] = np.sum(det_val)/len(wave_nums)
        else:
            det= -1* np.ones(len(dists))
        return det
