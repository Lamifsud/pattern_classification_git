import numpy as np 

def spk_count(df_spk, df_task_ts) : 

    n_neurons = Info_session.n_neurons
    n_trials = Info_session.n_trials
    n_periods = Info_session.n_periods

    spk_count_arr = np.zeros((n_neurons, n_trials, n_periods - 1))

    for neuron in range(n_neurons):
        for trial in range(n_trials):
                for per in range(n_periods - 1):

               
                    start = df_task_ts.columns[per]
                    stop = df_task_ts.columns[per+1]

                    # align to zero the start and end time of each period
                    t_start = df_task_ts[f'{start}'][trial] - df_task_ts['Touch_times'][trial]
                    t_stop = df_task_ts[f'{stop}'][trial] - df_task_ts['Touch_times'][trial]

                    # get spikes count between start and end of each period 
                    for spk in range(len(df_spk[trial][neuron])): 
                        if df_spk[trial][neuron][spk]>t_start and df_spk[trial][neuron][spk]<t_stop:
                            spk_count_arr[neuron][trial][per] += 1  
    return spk_count_arr


def spk_rate(Info_session, df_spk, df_task_ts):
    n_neurons = Info_session.n_neurons
    n_trials = Info_session.n_trials
    n_periods = Info_session.n_periods

    spk_count_arr = spk_count(Info_session, df_spk, df_task_ts)
    spk_rate_arr = np.zeros((n_neurons, n_trials, n_periods-1))
    
    for neuron in range(n_neurons):
        for trial in range(n_trials):
            for period in range(n_periods-1):
                per = df_task_ts.loc[trial][period]
                next_per = df_task_ts.loc[trial][period+1]
                # get the time interval in each period
                time = ((next_per-per) * 0.001)
                # compute the firing rate 
                
                spk_rate_arr[neuron][trial][period] = spk_count_arr[neuron][trial][period] / time
      
    return spk_rate_arr