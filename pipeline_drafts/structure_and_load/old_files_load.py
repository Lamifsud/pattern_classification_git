import scipy.io as sio
import os
import re
import numpy as np
from neo.core import SpikeTrain
from quantities import s



class Data:
    

    def __init__(self, spike_path, task_path):
        """
        Initialize the Data class with spike data and task data paths

        Args:
        - spike_path (str): Path to the directory containing spike data files.
        - task_path (str): Path to the task data file.

        Example usage:
        - data_instance = Data('spike_data_directory', 'task_data_file.mat')
        """
        self.spike_path = spike_path
        self.task_path = task_path

    def sort_by_prob(self):

        """
        Sorts contact (units) information by probe and handles encoding of contacts with multiple units.

        Returns:
        - units_by_prob (dict): A dictionary where keys are probe names and values are sorted contact indices.
        - file_paths (list of lists): A list of file paths grouped by probe.

        Example usage:
        - units_by_prob, file_paths = your_instance.sort_by_prob()
        """

        contacts_probes = [[],[]]
        units_probes = [[],[]]
        file_paths = [[],[]]
        units_by_prob = {}

        count_SUA = 0

        file_name = os.listdir(self.spike_path)

        for file in file_name:

            # get contact and unit indices from filename
            match_contact = re.search(r'contact(\d+)', file)
            match_unit = re.search(r'unit(\d+)', file)

            contact_idx = int(match_contact.group(1))
            unit_idx = int(match_unit.group(1))

            # store contact and unit indices by probe
            if 'probe1' in file:
                contacts_probes[0].append(contact_idx)
                units_probes[0].append(unit_idx)
                file_paths[0].append(os.path.join(self.spike_path, file))

            else:
                contacts_probes[1].append(contact_idx)
                units_probes[1].append(unit_idx)
                file_paths[1].append(os.path.join(self.spike_path, file))

        
        for prob_idx, (contact, unit) in enumerate(zip(contacts_probes, units_probes)):

            # sort contact and store in dict
            contacts_probes[prob_idx].sort()
            units_by_prob[f'prob {prob_idx}'] = contact

            # encode contact with more than one unit 
            for idx, val  in enumerate(unit) :
                #contact[idx] = f'unit {contact[idx]}' # if we want to get unit n° idx)
                if val>1:
                    count_SUA+=1
                    contact[idx-1] = str(contact[idx-1]) + 'a'
                    contact[idx] = str(contact[idx]) + chr(ord('a') + count_SUA)

            print(f'units probe {prob_idx} : {contacts_probes[prob_idx]}')

        return contacts_probes, file_paths


    def get_spike_times(self, split_probe):
        """
        Retrieves spike times from spike data files and optionally splits them by probe.

        Args:
        - split_probe (bool): If True, spike times are separated by probe. If False, they are concatenated.

        Returns:
        - spike_ts (list or list of lists): List of spike times. If split_probe is True, it's a list of lists; otherwise, a single list.
        """

        unit_by_probe, file_paths = self.sort_by_prob() 

        if split_probe == True :
            spike_ts = [[],[]]
            
            for prob_idx in range(len(file_paths)):
                for file in file_paths[prob_idx]:
                    spike_ts[prob_idx].append(sio.loadmat(file)['spikeData'][0][0][0])
            
            print(f'\nSplited in spike_ts = True :\n{len(spike_ts[0])} units in spike_ts[0]\n{len(spike_ts[1])} units in spike_ts[1]')

        
        else : 
            spike_ts = [[]]
            
            for prob_idx in range(len(file_paths)):
                for file in file_paths[prob_idx]:
                    spike_ts[0].append(sio.loadmat(file)['spikeData'][0][0][0])
            print(f'\nSplitted in spike_ts = False :\n{len(spike_ts[0])} units in spike_ts')
        
        return spike_ts, unit_by_probe

    def get_event_times(self):
        """
        Retrieves event times from task data file, sorts them, and creates a time vector.

        Returns:
        - event_labels (list): List of event labels.
        - event_times (numpy.ndarray): Array of event times.
        - event_times_vector (numpy.ndarray): Unique time vector containing all event times.

        Example usage:
        - event_labels, event_times, event_times_vector = your_instance.get_event_times()
        """

        task_data = sio.loadmat(self.task_path)

        # get keys list without '__header__', '__version__' and '__globals__'
        keys = list(task_data.keys())[3:-1] 

        # get event_labels labels 
        event_labels = []
        for index, key in enumerate(keys) : 
            if key.endswith("_times") :
                event_labels.append(key)

        # sort event_labels labels according task design  
        event_labels = [event_labels[-1],  event_labels[-3], event_labels[1], event_labels[2], event_labels[3], event_labels[0], event_labels[-2]]
        print(f'event labels : {event_labels}')

        # get times by event_labels
        times = []

        for i, event in enumerate(event_labels):
            if event == 'STOP_times':
                times.append(task_data[f'Target_times'][0] + 1000)
            else:
                times.append(task_data[f'{event}'][0])
            

        event_times = np.array(times, dtype=object)

        # get a unique time vector containing all event_labels for the session  
        event_times_vector = []
        for event in range(event_times.shape[1]):
            event_times_vector.append(event_times[:, event])
            
            
        event_times_vector = np.array(event_times_vector).flatten()

        print(f'\nevent_times shape : {event_times.shape}')
        print(f'\nevent_times_vector shape : {event_times_vector.shape}')

        return event_labels, event_times, event_times_vector







class Data2:
    

    def __init__(self, spike_path, task_path, split_probe):
        """
        Initialize the Data class with spike data and task data paths.

        Args:
        - spike_path (str): Path to the directory containing spike data files.
        - task_path (str): Path to the task data file.

        Example usage:
        - data_instance = Data('spike_data_directory', 'task_data_file.mat')
        """
        
        self.spike_path = spike_path
        self.task_path = task_path
        self.split_probe = split_probe

    def sort_by_prob(self):

        """
        Sorts contact (units) information by probe and handles encoding of contacts with multiple units.

        Returns:
        - units_by_prob (dict): A dictionary where keys are probe names and values are sorted contact indices.
        - file_paths (list of lists): A list of file paths grouped by probe.

        Example usage:
        - units_by_prob, file_paths = your_instance.sort_by_prob()
        """

        if self.split_probe == True : 
            contacts_probes = [[], []]
            units_by_probes = [[], []]
            file_paths = [[], []]

        else : 
            contacts_probes = [[]]
            units_by_probes = [[]]
            file_paths = [[]]

        units_by_prob = {}
        count_SUA = 0

        file_name = os.listdir(self.spike_path)

        for file in file_name:

            # get contact and unit indices from filename
            match_contact = re.search(r'contact(\d+)', file)
            match_unit = re.search(r'unit(\d+)', file)

            contact_idx = int(match_contact.group(1))
            unit_idx = int(match_unit.group(1))

            # store contact and unit indices by probe
            if self.split_probe == True :
                if 'probe1' in file:
                    contacts_probes[0].append(contact_idx)
                    units_by_probes[0].append(unit_idx)
                    file_paths[0].append(os.path.join(self.spike_path, file))

                else:
                    contacts_probes[1].append(contact_idx) 
                    units_by_probes[1].append(unit_idx)
                    file_paths[1].append(os.path.join(self.spike_path, file))

            else : 
                if 'probe1' in file:
                    contacts_probes[0].append('p1-' + str(contact_idx))
                    

                else:
                    contacts_probes[0].append('p2-' + str(contact_idx))
       
            units_by_probes[0].append(unit_idx)
            file_paths[0].append(os.path.join(self.spike_path, file))

  
        if  self.split_probe == True:
            for prob_idx, (contact, unit) in enumerate(zip(contacts_probes, units_by_probes)):

                # sort contact and store in dict
                contacts_probes[prob_idx].sort()
                units_by_prob[f'prob {prob_idx}'] = contact

                # encode contact with more than one unit 
                for idx, val  in enumerate(unit) :
                    #contact[idx] = f'unit {contact[idx]}' # if we want to get unit n° idx)
                    if val>1:
                        count_SUA+=1
                        contact[idx-1] = str(contact[idx-1]) + 'a'
                        contact[idx] = str(contact[idx]) + chr(ord('a') + count_SUA)

                print(f'units probe {prob_idx} : {contacts_probes[prob_idx]}')

        else : 
            #contact_sort_by_prob = [sorted(contacts_probes[0], key=lambda x: (int(x.split('-')[0].replace('p', '')), int(x.split('-')[1])))]
            
            for (contact, unit) in zip(contacts_probes, units_by_probes):

                # encode contact with more than one unit 
                for idx, val  in enumerate(unit) :
                    #contact[idx] = f'unit {contact[idx]}' # if we want to get unit n° idx)
                    if val>1:
                        count_SUA+=1
                        #contact[idx-1] = str(contact[idx-1]) + 'a'
                        contact[idx] = str(contact[idx]) + chr(ord('a') + count_SUA)
                
                        multi_contact_to_encode = str(contact[idx])

                        for ele in contacts_probes:
                            if ele == multi_contact_to_encode :
                                contacts_probes[ele] = str(contact[idx]) + 'a'


                print()
                # sort contact and store in dict
                #contacts_probes = [sorted(contacts_probes[0], key=lambda x: (int(x.split('-')[1]), x))]
                #units_by_prob[f'prob 01'] = contacts_probes[0]

                #print(f'units probe 01 : {contacts_probes[0]}')


        return contacts_probes, units_by_probes, units_by_prob,  file_paths
