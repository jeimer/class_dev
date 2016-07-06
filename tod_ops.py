import numpy as np

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
