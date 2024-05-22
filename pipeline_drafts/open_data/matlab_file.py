import os
import scipy.io as sio 
import pandas as pd
import numpy as np

def from_lab(lab_desktop, session) :

    if lab_desktop : 
        data_path = f'/home/INT/mifsud.l/Bureau/data/Tommy/{session}/modified_data/' #f'~/Bureau/data/Tommy/{session}/modified_data'
        info_path = '/home/INT/mifsud.l/Bureau/Lists&Documentation/TomyCerebusSpikes_Updated_June2023.xlsx'
        result_path = '~/Bureau/results'

    else : 
        data_path = f'/home/laurie/Bureau/pattern_classification/data/Tommy_new/{session}/modified_data/'
        info_path = '/home/laurie/Bureau/pattern_classification/data/Tommy_new/session_info.xlsx'

    return session, data_path, info_path

def open(path):
    """
    Open a matlab behaviour structure from OrganizeBehaviour and save it in a python dictionary. One
    SESSION corresponds to one behavioural file - independently of the number of probes.
    :param filename: name of the matlab structure containing the behavioural information
    :return: behaviour: dictionary containing the same fields as the behaviour structure from matlab
    """
    # initialize a counter for the number of files loaded
    n_file = -1
    # get filenames in list in the session directory
    filenames = os.listdir(path)

    data = [] 
    load_info = []
    
    # Loop through MATLAB files in the session directory
    for matfile in filenames:

        n_file += 1  
        #print(f'loading {matfile}')
        load_info.append((n_file, matfile))

        # get the full file path for the current file and load MATLAB file
        filepath = os.path.join(path, matfile) 
        matfile = sio.loadmat(filepath)

        # get the name of the sub-structure within the MATLAB file
        data_str_name = list(matfile.keys())[-1]

        # get the field names within the sub-structure and use them as keys for the new dictionary
        fields = []
        for key in matfile[data_str_name].dtype.fields.keys():
            fields.append(key)

        # Extract the data inside the sub-structure and save dictionnary in list
        data_ = matfile[data_str_name][0][0]
        data.append({field: data_[i_field][:, 0] for i_field, field in enumerate(fields)})
        

    print(f'\n{n_file + 1} files loaded')
    
    return data, load_info


def clean(info_path):
    """
    Clean the data from an Excel file containing information about behavioural data.
    
    Parameters:
    -----------
    info_path : str
        Path to the Excel file containing the information.

    Returns:
    --------
    df : pandas.DataFrame
        Cleaned DataFrame containing the behavioural information.
    """
    # Read the Excel file into a pandas DataFrame
    df = pd.read_excel(info_path)
    
    # Drop rows where all values are NaN
    df.dropna(how='all', inplace=True)
    
    # Drop unnecessary columns
    df.drop(columns=['BehDir', 'pitch', 'A/P', 'Lat', 'depth', 'SNR', 'Unnamed: 14', 'depth (no buffer)'], inplace=True)
    
    # Rename column
    df.rename(columns={'%Plexon_spike_file' : 'Plexon_spike_file'}, inplace=True)
    
    # Convert float columns to integer
    float_column_names = df.select_dtypes(include=['float']).columns
    df[float_column_names] = df[float_column_names].astype(int)
    
    return df

def extract_data(load_info, data, target_keys_OFF, target_keys_ON,  event_keys_OFF):
    """
    Extract spike times from the given data.

    Parameters:
    -----------
    load_info : list of tuples
        List containing information about loaded files, where each tuple is in the format (n_file, matfile).
    data : list of dictionaries
        List containing data dictionaries for each unit.

    Returns:
    --------
    spike_times : list of tuples
        List containing tuples of (loading info, spike times) for each unit.
    """
    spike_times = []
    for unit_idx in range(len(data)):
        spike_times.append((load_info[unit_idx], data[unit_idx]['ts']))  

    task_times = []
    for unit_idx in range(len(data)):
        times_keys = [key for key in data[unit_idx] if key != 'ts' and key not in target_keys_OFF and key not in event_keys_OFF]
        task_info = {}
        for key_time in times_keys:
            task_info[key_time] = data[unit_idx][key_time]
        task_times.append((load_info[unit_idx], task_info))

    target_info = []
    for unit_idx in range(len(data)):
        target_ = {}
        for key_target in target_keys_ON:
            target_[key_target] = data[unit_idx][key_target]
        target_info.append((load_info[unit_idx], target_))

    return spike_times, task_times, target_info



def get_event_labels(task_data):
    """
    Get event labels from the given task data.

    Parameters:
    -----------
    task_data : list of tuples
        List containing tuples of (loading info, task data dictionary) for each unit.

    Returns:
    --------
    event_labels : list
        List of event labels extracted from task data.
    """
    event_labels = list(task_data[0][1].keys())
    return event_labels


# get event times by trial for each neuron
def get_event_times(task_data, event_labels):
    """
    Retrieves event times from task data file sorts them for each unit

    Args:
    - task_data (list): List of tuples where each tuple contains file information and a dictionary of task-related data.

    Returns:
    - event_times (list of numpy.ndarray): List of arrays containing event times.

    Example usage:
    - event_labels, event_times, event_times_vector = event_times(task_data)
    """

    # Get event times by event labels
    event_times = []
    for unit_idx in range(len(task_data)):
        times = []
        for event in event_labels:
            times.append(task_data[unit_idx][1][event])
        event_times.append((task_data[unit_idx][0], np.array(times, dtype=object)))

    return event_times