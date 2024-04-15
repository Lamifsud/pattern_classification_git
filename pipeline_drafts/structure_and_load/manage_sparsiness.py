#import re
# verify times stamps across neurons 

# event_times[0] unit 0
# event_times[0][0] --> info[0] = loading index, info[1] = matfile
# event_times[0][1] --> data[event][trials]

'''for i in range(len(event_times)) : 
    file = event_times[i][0][1]
    shape = event_times[i][1].shape

    # get contact and unit indices from filename
    match_probe = re.search(r'probe(\d+)', file)
    match_contact = re.search(r'contact(\d+)', file)
    match_unit = re.search(r'unit(\d+)', file)

    probe_idx = int(match_probe.group(1))
    contact_idx = int(match_contact.group(1))
    unit_idx = int(match_unit.group(1))

    sub_data = info_session[(info_session['probe'] == probe_idx) & (info_session['channel'] == contact_idx) & (info_session['unit'] == unit_idx)]

    print(file)

    print(sub_data[['start','stop', 'elitrials']])
    print(shape)
    print(f'{event_times[i][1][0][0]}, {event_times[i][1][-1][-1]}\n')'''


# check trials across neurons
'''
##look for units showing a patterns in info_session ##

# same bloc / without elitrials 
id_1 = 3
id_2 = 4
id_3 = 9
# same bloc / with elitrials 
id_4 = 34
# bloc 4 / 8 - elitrials [55,69] [129,139]
id_5 = 12
# bloc 9/11 - elitrials [[140, 144]]
id_6 = 22

units_list = [id_1, id_2, id_3, id_4, id_5, id_6]

print(spike_times[id_1][0], df_task_ts_by_neuron[id_1][1].shape)
print(spike_times[id_2][0], df_task_ts_by_neuron[id_2][1].shape)
print(spike_times[id_3][0], df_task_ts_by_neuron[id_3][1].shape)
print(spike_times[id_4][0], df_task_ts_by_neuron[id_4][1].shape)
print(spike_times[id_5][0], df_task_ts_by_neuron[id_5][1].shape)
print(spike_times[id_6][0], df_task_ts_by_neuron[id_6][1].shape)

idx_trial = df_task_ts[df_task_ts[f'{event}'] == t_start].index[0]




'''




#split spike_times by period ? 
'''n_units = len(spike_times)

for unit in range(4):
    filename = spike_times[unit][0][1]
    print(filename)
    # get contact and unit indices from filename
    match_probe = re.search(r'probe(\d+)', filename)
    match_contact = re.search(r'contact(\d+)', filename)
    match_unit = re.search(r'unit(\d+)', filename)

    probe_idx = int(match_probe.group(1))
    contact_idx = int(match_contact.group(1))
    unit_idx = int(match_unit.group(1))

    sub_data = info_session[(info_session['probe'] == probe_idx) & (info_session['channel'] == contact_idx) & (info_session['unit'] == unit_idx)]

    print(sub_data[['probe', 'channel' , 'unit']])
    print('\n')

    idx_min_trial = df_task_ts_by_neuron[unit][1].index[0]
    idx_max_trial = df_task_ts_by_neuron[unit][1].index[-1]
    idx_trials = np.arange(idx_min_trial, idx_max_trial + 1) 

    elitrials = sub_data['elitrials'].values[0]

    if elitrials is not None : 
        print(elitrials)
        # Initialize an empty array to store the result
        list_elitrials = np.array([])

        for trial_range in range(len(elitrials)):
            start = elitrials[trial_range][0]
            stop = elitrials[trial_range][-1]           
            
            # Create a range of values and concatenate it to the result
            list_elitrials = np.concatenate((list_elitrials, np.arange(start, stop + 1)))

        print(list_elitrials)

        # create the index list 
        print(idx_trials)
        print(f'{idx_trials.shape} - {len(list_elitrials)}') 

        # exclude elitrials 
        idx_trials = idx_trials[~np.isin(idx_trials, list_elitrials)]
        print(idx_trials)'''