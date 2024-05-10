import numpy as np
# get spike times vector (t0-n0, t0-n1, ... tn-nk) : 
def get_spike_times_vect(spikes_times_aligned):
    n_trials = spikes_times_aligned.shape[1]
    n_neurons = spikes_times_aligned.shape[0]
    spk_times = np.zeros((n_trials, n_neurons), dtype=object)

    for n in range (n_neurons):
        for t in range(n_trials):
                spk_times[t,n] = spikes_times_aligned[n][t]['spike_time']

    spike_times_vec = np.concatenate(spk_times.flatten())

    return spike_times_vec