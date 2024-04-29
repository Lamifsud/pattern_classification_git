import numpy as np 

def get_max_length_trials(data):
    n_trials = data.shape[0]
    time_ = np.zeros((n_trials), dtype=int)
    for t in range(n_trials): 
        time_[t] = data[t]['t_stop_aligned']
    max_length = np.max(time_)

    return max_length


def get_max_length_ISI(n_trials, ISI_distance):
    time_ = np.zeros((n_trials), dtype=int)
    for t in range(n_trials): 
        time_[t] = ISI_distance[t].shape[1]
    max_length = np.max(time_)

    return max_length, time_