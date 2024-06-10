import numpy as np 
import scipy.ndimage as spn 

def binarize_spike_times(spikes_times_aligned, max_length):
    n_neurons, n_trials = spikes_times_aligned.shape
    spikes_binarized = np.zeros((n_trials, max_length, n_neurons), dtype='float32')
    list_nan = []
    for n in range(n_neurons):
        for t in range(n_trials):
            spiketrain = spikes_times_aligned[n][t]['spike_train']
            t_start = spikes_times_aligned[n][t]['t_start_ref'] 
            t_stop = spikes_times_aligned[n][t]['t_stop_aligned'] 

            if len(spiketrain) > 0 and spiketrain[0] != 0:
                spk = spiketrain.magnitude - t_start
                spikes_binarized[t, spk, n] = 1 
            else : 
                spikes_binarized[t,:,n] = 0 
            
            spikes_binarized[t, t_stop:max_length, n] = np.nan

    
    return spikes_binarized


def convolve_spike_binarized(spikes_binarized, sigma):
    n_sigma = len(sigma)
    delta_time = 1
    sigma_dt = sigma * delta_time 
    n_trials, n_times, n_neurons = spikes_binarized.shape

    spike_convolved = np.zeros((n_trials, n_times, n_neurons))

    for idx, s in enumerate(sigma):
        scaling = sigma_dt[idx] * np.sqrt(2*np.pi)
        for t in range(n_trials) : 
            for n in range(n_neurons):
                smoothed_spk = spn.gaussian_filter1d(spikes_binarized[t,:,n], sigma_dt[idx]) * scaling
                spike_convolved[t,:,n] = smoothed_spk
    
    return spike_convolved