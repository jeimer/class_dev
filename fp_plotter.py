import numpy as np

class FpModule():
    def __init__(self, num_rows=1, pitch=1):
        self._num_hex_rows = num_rows
        self._pitch = pitch
        self._centers = []
        self._mods = []
        self.make_hex_module(num_rows)
        return

    def row_num_check(self, num):
        if num%2 == 0:
            raise ValueError('The number of rows must be odd')

    def mod_append(self, new):
        if len(self._centers) == 0:
            self._centers = new
        else:
            self._centers = np.append(self._centers, new, 0)
        return

    def add_row(self, num):
        num_in_row = self._num_hex_rows - np.abs(num)
        new = np.ones([num_in_row,2])
        new[:, 1] = np.sqrt(3) * num * self._pitch/2.
        xc = np.arange(num_in_row) - (num_in_row - 1)/2.
        new[:,0] = -xc * self._pitch
        self.mod_append(new)
        return

    def make_hex_module(self, num_hex_rows):
        try:
            self.row_num_check(num_hex_rows)
        except ValueError:
            print('The number of rows must be odd')
            raise
        self._centers = []
        self._num_hex_rows = num_hex_rows
        rows = np.arange(num_hex_rows) - (num_hex_rows - 1)/2
        for row in rows:
            self.add_row(row)
        return

    def rotate_module(self, ang):
        for pix in range(len(self._centers)):
            tx, ty = self._centers[pix,:]
            self._centers[pix, 0] = tx * np.cos(ang) - ty * np.sin(ang)
            self._centers[pix, 1] = tx * np.sin(ang) + ty * np.cos(ang)
        return

    def remove_pix(self, pos):
        '''
        removes the specified positions from the self._mod_centers list.
        '''
        self._centers = np.delete(self._centers, pos, 0)
        return

    def set_centers(self, centers):
        self._centers = centers

    def centers(self):
        return self._centers

class FocalPlane():
    def __init__(self, recv = 'q_band'):
        self._recv = recv
        self._num_mods = 0
        self._mod_centers = []
        self._centers = np.transpose(np.array([[],[]]))
        self._mods = []
        self._fp_maker = {'q_band':self.q_array,'w_band':self.w_array,
                          'hf':self.hf_array}
        self._pix_scale = 1  #approx ratio of size of pixel to full array
        self._fp_maker[recv]()


    def append_module(self, center, rot, module):
        module.rotate_module(rot)
        module.set_centers(module.centers() + center)
        self._mods += [module]
        self._centers =  np.vstack([self._centers, module.centers()])
        self._mod_centers += [center]
        self._num_mods += 1
        return

    def q_array(self):
        mod1 = FpModule(7)
        mod1.remove_pix(np.hstack([np.arange(4), np.arange(22,37)]))
        mod2 = FpModule(7)
        mod2.remove_pix(np.hstack([np.arange(0,15), np.arange(33,37)]))
        self.append_module([0, -1./2.], 0, mod1)
        self.append_module([0, 1./2.], 0, mod2)
        self._pix_scale = 1./7.
        return

    def centers(self):
        return self._centers

    def set_centers(self, centers):
        self._centers = centers
        return

    def w_array(self):
        return

    def hf_array(self):
        return


class FpPlotter():
    def __init__(self, f, fp = FocalPlane()):
        self._fig = f
        self._figsize = f.get_size_inches()
        self._fp = fp
        self._scale = 1
        self._ax_size = np.ones(2) * fp._pix_scale * .9
        self.scale_fp()
        return

    def scale_fp(self):
        fp_x = self._fp.centers()[:,0].max() * 2
        fp_y = self._fp.centers()[:,1].max() * 2
        scale = np.min([self._figsize[0]/fp_x, self._figsize[1]/fp_y])
        self._scale = scale
        self._fp.set_centers(0.9 * np.array([self._fp.centers()[:,0] / self._figsize[0],
                                             self._fp.centers()[:,1] / self._figsize[1]]).T)
        self._fp.set_centers(self._fp.centers() *  scale + np.array([.5,.5]))
        return

    def det_plot(self, center):
        corner = center - self._ax_size/2.
        ax = self._fig.add_axes([corner[0], corner[1],
                                 self._ax_size[0], self._ax_size[1]])
        return ax

    def pick_plot(self, det_uid):




def test(f):
    size = f.get_size_inches()
    return size
