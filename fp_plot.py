#!/usr/bin/evn

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np

class plt_fp(object):
    ''' motivated by John A.'s focal plane plotting tool, but have not been able to import into omar's jupyter notebook
    The constructor requires a mapping between detector number, H,V,DD, DS, column number, row number, wafer name, elevation offsets,
    azimuth offsets, and detector rotation angle.

    The class will load the array mappings contained within the depots defined in the user's .moby2 file'''

    def __init__(self, array_path, band = 'q'):
        '''array_path: path to the folder containing images of the array and current mapping of detectors
        band: either q, w1, w2, or hf'''
        self._band = band.lower()
        self._array_path = array_path
        self.load_array_data()
        return

    def load_array_data(self):
        '''for now this function hard codes the path to ArrayData/current/array_data.txt for focal planes. It would be
        better to strip this info directly from .moby2 file'''
        if self._band == 'q':
            self._mapfile = array_path + '/array_data.txt'
            self._baseplate = mpimg.imread(arra_path + '/interface_plate_v1.png')
        return
