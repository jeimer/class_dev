
import emcee
import numpy as np



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

