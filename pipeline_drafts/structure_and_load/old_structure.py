import numpy as np
import pandas as pd
from neo.core import Event
from quantities import s
import numpy as np

# to go further with NEO : https://neo.readthedocs.io/en/stable/grouping.html


def events_by_trial(event_times, event_labels):  
    '''
    Structure periods timestamps by trial and store in a data frame.

    Args:
    - event_times (numpy.ndarray): Array of event times where rows represent events, and columns represent trials.
    - event_labels (list): List of event labels corresponding to the columns of the data frame.

    Returns:
    - trials_ts (numpy.ndarray): Array of event times structured by trial.
    - df_task_ts (pandas.DataFrame): Data frame containing event times structured by trial.

    '''

    n_events = event_times.shape[0]
    n_trials = event_times.shape[1]

    print(f'n_trials = {n_trials}\nn_events per trial = {n_events}')

    trials_ts = np.zeros((n_trials, n_events), dtype=int)
    trials_ts_aligned = np.zeros((n_trials, n_events), dtype=int)
    events = []
    index_trial = np.arange(n_trials)

    # split time stamps for each period according to the trial
    for trial in range(n_trials) : 
        for event in range(n_events):
                start_time_trial = event_times[0][trial]
                evt_time = event_times[event][trial]
                trials_ts[trial][event] = evt_time
                trials_ts_aligned[trial][event] = evt_time - start_time_trial
        
        events.append(Event(trials_ts[trial]*s, labels=event_labels, dtype='U'))

    df_task_ts = pd.DataFrame(trials_ts_aligned, columns=event_labels, index=index_trial)

    return trials_ts, trials_ts_aligned, df_task_ts, events



def spike_by_trial(trials_ts, spike_times, unit_labels, split_probe):  
    '''split the spike time vector by trial RUN WITH EVENT TIMES NOT ALIGNED'''

    list_spk = []
    list_df_spk = []

    if split_probe == True:
        n_probes = len(spike_times)

        for idx_probe in range(n_probes):
            list_spk.append([])
            n_neurons = len(spike_times[idx_probe])
       
            for idx_neuron in range(n_neurons):
                list_spk[idx_probe].append([])
                n_trials = trials_ts.shape[0]

                for trial in range(n_trials):    
                    # define the start and end time of each trial
                    t_start = trials_ts[trial,0]
                    t_stop = trials_ts[trial,-1]
                    
                    # get spikes between start and end of trial 
                    spk_tmp = spike_times[idx_probe][idx_neuron] 
                    sel_spk = np.logical_and(spk_tmp>t_start, spk_tmp<t_stop)
                    
                    # for trials without spikes 
                    if spk_tmp[sel_spk].shape[0] == 0:
                        list_spk[idx_probe][idx_neuron].append([])

                    else :
                        spk_ts_trial = spk_tmp[sel_spk] 
                        # fill the matrice with spike times aligned to 0
                        list_spk[idx_probe][idx_neuron].append(spk_ts_trial - t_start)

          
            
            df_spk = pd.DataFrame(list_spk[idx_probe], index=unit_labels[idx_probe])
            list_df_spk.append(df_spk)

    else:
        n_neurons = len(spike_times)

        for idx_neuron in range(n_neurons):
            list_spk.append([])
            n_trials = trials_ts.shape[0]

            for trial in range(n_trials):    
                
                # define the start and end time of each trial
                t_start = trials_ts[trial,0]
                t_stop = trials_ts[trial,-1]
                
                # get spikes between start and end
                spk_tmp = spike_times[idx_neuron]
                sel_spk = np.logical_and(spk_tmp>t_start, spk_tmp<t_stop)

                # for trials without spikes 
                if spk_tmp[sel_spk].shape[0] == 0:
                    list_spk[idx_neuron].append([])

                else :
                    spk_ts_trial = spk_tmp[sel_spk] 
                    # fill the matrice with spike times aligned to 0
                    list_spk[idx_neuron].append(spk_ts_trial - t_start)

                
        df_spk = pd.DataFrame(list_spk, index=unit_labels)
        
        list_df_spk = df_spk

    return list_df_spk, list_spk

def spike_by_trial(trials_ts, spike_times, unit_labels, split_probe):  
    '''split the spike time vector by trial RUN WITH EVENT TIMES NOT ALIGNED'''

    list_spk = []
    list_df_spk = []

    n_probes = len(spike_times)

    for idx_probe in range(n_probes):
        list_spk.append([])
        n_neurons = len(spike_times[idx_probe])
    
        for idx_neuron in range(n_neurons):
            list_spk[idx_probe].append([])
            n_trials = trials_ts.shape[0]

            for trial in range(n_trials):    
                # define the start and end time of each trial
                t_start = trials_ts[trial,0]
                t_stop = trials_ts[trial,-1]
                
                # get spikes between start and end of trial 
                spk_tmp = spike_times[idx_probe][idx_neuron] 
                sel_spk = np.logical_and(spk_tmp>t_start, spk_tmp<t_stop)
                
                # for trials without spikes 
                if spk_tmp[sel_spk].shape[0] == 0:
                    list_spk[idx_probe][idx_neuron].append([])

                else :
                    spk_ts_trial = spk_tmp[sel_spk] 
                    # fill the matrice with spike times aligned to 0
                    list_spk[idx_probe][idx_neuron].append(spk_ts_trial - t_start)

        
        
        df_spk = pd.DataFrame(list_spk[idx_probe], index=unit_labels[idx_probe])
        list_df_spk.append(df_spk)

    return list_df_spk, list_spk

def align(list_spk, trials_ts): 
    n_events = trials_ts.shape[0]
    n_trials = trials_ts.shape[1]


    events = []
    index_trial = np.arange(n_trials)

    # split time stamps for each period according to the trial
    for trial in range(n_trials) : 
        for event in range(n_events):
                #evt_time = event_times[event][trial] - event_times[0][trial]
                trials_ts[trial][event] = trials_ts[event][trial]
        
        events.append(Event( trials_ts[trial]*s, labels=trials_ts, dtype='U'))

    df_task_ts = pd.DataFrame(trials_ts, columns=trials_ts, index =index_trial)

    return trials_ts, df_task_ts, events

"""def spike train (Info_session, trials_ts, spk_ts_trial):
    n_neurons = Info_session.n_neurons
    n_trials = Info_session.n_trials
    n_periods = Info_session.n_periods

    spk_ts_trial_aligned = []

    for n in range(n_neurons) : 
        spk_ts_trial_aligned.append([])

        for trial in range(n_trials) :
            spk_ts_trial_aligned[n].append([])
    
            for per in range (n_periods - 1):
                t_start = trials_ts[trial, per]
                t_stop = trials_ts[trial, per+1]
                n_times = (t_stop - t_start)
                spk_ts_trial_aligned[n][trial].append(np.linspace(0, 0, num=n_times+1,  dtype=float))
        
                for spk in range(len(spk_ts_trial[n][trial])):
                    if (t_start <= spk_ts_trial[n][trial][spk] <= t_stop) :
                        spk_aligned = spk_ts_trial[n][trial][spk] - t_start
                        spk_ts_trial_aligned[n][trial][per][spk_aligned] = 1

    spk_aligned = np.array(spk_ts_trial_aligned, dtype=object)

    return spk_aligned"""
    

def align_time(Info_session, trials_ts):
    n_trials = Info_session.n_trials
    n_periods = Info_session.n_periods

    'align times of each trials to 0'

    times_aligned = []
    
    for trial in range(n_trials) :
        times_aligned.append([])
    
        for per in range (n_periods-1):
        
            t_start = trials_ts[trial,per]
            t_stop = trials_ts[trial,per+1]
            n_times = (t_stop - t_start)
            times_aligned[trial].append(np.linspace(t_start, t_stop, num=n_times+1, dtype=int))

    times_aligned = np.array(times_aligned, dtype=object)

    return times_aligned