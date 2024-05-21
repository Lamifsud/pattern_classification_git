import numpy as np
# get spike times vector (t0-n0, t0-n1, ... tn-nk) : 
def get_spike_times_vect(spikes_times_aligned):
    n_trials = spikes_times_aligned.shape[1]
    n_neurons = spikes_times_aligned.shape[0]
    spk_times = np.full((n_trials, n_neurons), np.nan(), dtype=object)
    last_spike = np.zeros((n_trials, n_neurons), dtype=int)

    for n in range (n_neurons):
        for t in range(n_trials):
                spk_times[t,n] = spikes_times_aligned[n][t]['spike_time']
                if spikes_times_aligned[n][t]['spike_time'].shape[0] > 0:
                    last_spike[t,n] = spikes_times_aligned[n][t]['spike_time'].max()

    spike_times_vec = np.concatenate(spk_times.flatten())

    return spike_times_vec, last_spike


def create_epochs(spikes_times_aligned): 
    n_trials = spikes_times_aligned.shape[1]
    epochs = np.zeros((n_trials, 2), dtype=int)

    for t in range(n_trials):
            t_start = int(spikes_times_aligned[0][t]['t_start_aligned'])
            t_stop = int(spikes_times_aligned[0][t]['t_stop_aligned'])
            epochs[t] = t_start, t_stop
            
    return epochs