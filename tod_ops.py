import numpy as np

def data_valid_edges(tod):
    '''function returns the indecies of the edges of a tod.data array where the tes data are non-zero.
    tod: valid moby2 tod object.'''
    array_on_index = np.where(tod.data[0])[0]
    off_edge = np.where((array_on_index[1:]-1) - array_on_index[:-1])[0]
    inner_edges = []
    for element in off_edge:
        inner_edges += [array_on_index[element]]
        inner_edges += [array_on_index[element+1]]
    return np.hstack((array_on_index[0],inner_edges, array_on_index[-1]))
