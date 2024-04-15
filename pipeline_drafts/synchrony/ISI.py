import numpy as np 

def compute(x):
    n_times = x['t_stop']
    x_isi_in_time = np.zeros(n_times)

    # Utiliser les valeurs d'`x['spike_time']` comme indices de début et de fin pour chaque intervalle entre les spikes
    start_indices = x['spike_time'][:-1]
    end_indices = x['spike_time'][1:]

    # Mettre à True les indices entre chaque paire d'indices dans mask
    for start, end in zip(start_indices, end_indices):
        x_isi_in_time[start:end] = np.repeat(end-start, end-start)

    return x_isi_in_time

def distance(x_isi, y_isi):
    ISI_distance = np.zeros_like(x_isi)
    ISI_distance[np.logical_and(x_isi==0, y_isi>0)] = -1 
    ISI_distance[np.logical_and(x_isi>0, y_isi==0)] = 1 

    # indices where x_isi is smaller than y_isi 
    idx_xsy = x_isi < y_isi
    ISI_distance[idx_xsy] =  (x_isi[idx_xsy] / y_isi[idx_xsy]) - 1

    # indices where y_isi is smaller than x_isi 
    idx_ysx = y_isi < x_isi
    ISI_distance[idx_ysx] = -((y_isi[idx_ysx] / x_isi[idx_ysx]) - 1)

    return ISI_distance