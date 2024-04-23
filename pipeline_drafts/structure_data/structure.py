import re
import pandas as pd
import numpy as np
from neo.core import Event
from quantities import s 
from quantities import millisecond as ms

def info(load_info, df, session):
    """
    Extract information about probes, contacts, and units from filenames and match them with corresponding data from an Excel file.
    
    Parameters:
    -----------
    load_info : list of tuples
        List containing information about loaded files, where each tuple is in the format (n_file, matfile).
    df : pandas.DataFrame
        DataFrame containing information about behavioural data, typically loaded from an Excel file.
    session : str
        Identifier for the session.

    Returns:
    --------
    info_units : list of dictionaries
        List of dictionaries containing information about probes, contacts, and units extracted from filenames.
    info_session : pandas.DataFrame
        DataFrame containing information about probes, contacts, and units matched with corresponding data from the Excel file.
    """
    # Initialize lists to store probe, contact, and unit information
    list_prob = []
    list_chan = []
    list_unit = []

    info_units = []

    # Extract probe, contact, and unit information from filenames and store in dictionaries
    for i in range(len(load_info)) :
        matfile = load_info[i][1]
        match_probe = re.search(r'probe(\d+)', matfile)
        match_contact = re.search(r'contact(\d+)', matfile)
        match_unit = re.search(r'unit(\d+)', matfile)

        info_units.append({
            'probe': int(match_probe.group(1)), 
            'contact': int(match_contact.group(1)), 
            'unit': int(match_unit.group(1))
        })

        list_prob.append(int(match_probe.group(1)))
        list_chan.append(int(match_contact.group(1)))
        list_unit.append(int(match_unit.group(1)))

    # Filter data from the Excel file based on session identifier and probe, contact, and unit information
    sub_info_session = df[df['Plexon_spike_file'].str.startswith(f'{session}')]

    info_session = pd.DataFrame()
    list_idx = []

    # Match probe, contact, and unit information with corresponding data from the Excel file
    for i in range(len(list_unit)):
        sub_data = sub_info_session[(sub_info_session['probe'] == list_prob[i]) & 
                                    (sub_info_session['channel'] == list_chan[i]) & 
                                    (sub_info_session['unit'] == list_unit[i])]
        list_idx.append(sub_data.index[0])
        info_session = pd.concat((info_session, sub_data))

    return info_units, info_session


def elitrials(df):
    """
    Convert a string representing Elite trials (in the format '[start:end start:end ...]') into a list of lists.
    
    Parameters:
    -----------
    entry : str
        String representing Elite trials.

    Returns:
    --------
    result : list of lists or None
        List of lists containing start and end pairs of Elite trials, or None if the entry is NaN.
    """
    if pd.notna(df):  # Check for NaN
        # Remove square brackets and split by space to separate pairs
        pairs = df.replace('[', '').replace(']', '').split()
        result = []
        for pair in pairs:
            start, end = map(int, pair.split(':'))
            result.append([start, end])
        return result
    else:
        return None


def CompleteTasktime(info_session, load_info, session):
    """
    Process information from info_session DataFrame.

    Parameters:
    -----------
    info_session : pandas.DataFrame
        DataFrame containing information about session data.
    load_info : list of tuples
        List containing information about loaded files, where each tuple is in the format (n_file, matfile).
    session : str
        Identifier for the session.

    Returns:
    --------
    None
    """
    # Calculate the first and last bloc start/stop times
    first_bloc = info_session['start'].min()
    last_bloc = info_session['stop'].max()

    # Print the start and stop times
    #print(f'start / stop:\n   {first_bloc} /   {last_bloc}')

    # Filter tasktime data where start and stop times match the first and last blocs and elitrials is NaN
    tasktime = info_session[(info_session['start'] == first_bloc) & 
                            (info_session['stop'] == last_bloc) & 
                            (info_session['elitrials'].isna())]

    # Check if there is any complete tasktime
    tasktimeComplete = tasktime.shape[0] != 0

    # If there is complete tasktime
    if tasktimeComplete:
        # Get unit information
        unit_info = tasktime[['probe', 'channel', 'unit']].values[0]
        print(f"{tasktime[['start', 'stop', 'elitrials']]} \n\nunit_info : {unit_info}")

        # Construct matfile name
        matfile = f'{session}_probe{unit_info[0]}_contact{unit_info[1]}_unit{unit_info[2]}.mat'

        # Find the loading index of the complete unit
        for i in load_info:
            if i[1] == matfile:
                completeUnit = i[0]
                print(f"{i[1]}\n\nunit's loading index = {completeUnit}")

        return completeUnit

    # If there is no complete tasktime
    else:
        print('ANY UNIT RECORDED OVER ENTIRE SESSION') 
        print(f"check here to construct session times by hand :\n {info_session['start'] == first_bloc}")


def events_by_trial(event_times, event_labels):
    '''
    Structure periods timestamps by trial and store in a data frame.

    Args:
    - event_times (numpy.ndarray): Array of event times where rows represent events, and columns represent trials.
    - event_labels (list): List of event labels corresponding to the columns of the data frame.

    Returns:
    - trials_ts (list of numpy.ndarray): List of event times structured by trial.
    - df_task_ts (list of pandas.DataFrame): List of data frames containing event times structured by trial.
    - events (list of list): List of Event objects.

    '''
    trials_ts = []  # List to store event times structured by trial
    df_task_ts_by_neuron = []  # List to store data frames containing event times structured by trial
    events = []  # List to store Event objects

    for unit_idx in range(len(event_times)):
        n_events = event_times[unit_idx][1].shape[0]
        n_trials = event_times[unit_idx][1].shape[1]

        events_ts = np.zeros((n_trials, n_events), dtype=int)
        event_list = []

        # Split time stamps for each period according to the trial
        for trial in range(n_trials):
            for event in range(n_events):
                events_ts[trial][event] = event_times[unit_idx][1][event][trial]
            
            event_list.append(Event(events_ts[trial] * ms, labels=event_labels, dtype='U'))

        events.append((event_times[unit_idx][0], event_list))
        trials_ts.append((event_times[unit_idx][0], events_ts))
        task_ts = pd.DataFrame(events_ts, columns=event_labels)
        df_task_ts_by_neuron.append((event_times[unit_idx][0], task_ts))

    return trials_ts, df_task_ts_by_neuron, events







def process_unit_trials(df_task_ts_by_neuron, df_task_ts, event_labels):
    """
    Process trials for each neuron.

    Parameters:
    -----------
    df_task_ts_by_neuron : list of tuples
        List containing tuples of (neuron index, DataFrame of task times) for each neuron.
    df_task_ts : pandas.DataFrame
        DataFrame containing task times.
    event_labels : list
        List of event labels.

    Returns:
    --------
    None
    """

    for unit in range(len(df_task_ts_by_neuron)):
        list_idx_trial = []
        times_unit = df_task_ts_by_neuron[unit][1]
        n_trials_unit = times_unit.shape[0]

        # Iterate over trials for the current unit
        for trial in range(n_trials_unit):
            for i, event in enumerate(event_labels[:-1]):
                t_start = times_unit.loc[trial][f'{event}']
                t_stop = times_unit.loc[trial][f'{event_labels[i+1]}']
                idx_trial = df_task_ts[df_task_ts[f'{event}'] == t_start].index[0]

            list_idx_trial.append(idx_trial)
        
        # Insert 'idx_ref_trial' column into times_unit DataFrame
        df_task_ts_by_neuron[unit][1].insert(0, 'idx_ref_trial', list_idx_trial)

    return df_task_ts_by_neuron
