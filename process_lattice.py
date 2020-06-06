# DONE

import numpy as np
from PLS.utils import *
import pandas as pd

'''
This is the first stage in the formation of the phone lattice.
The time frames not satisfying the hspike threshold will be dropped.

Input:  lstm_log_prob (LSTM log probabilities)
        h_spike (hyperparameter for phone lattice formation)

Output: new_lattice (phone lattice formed after this process)
        time_frames (retained time frames)

NOTE:   lstm_log_prob: nxT
        h_spike: values ranges from 0.1 to 0.5
        new_lattice: nxt where t<<T
        time_frames: 1D list with values which specifies the time frames that are retained

'''

def process_lattice_hspike(lstm_log_prob, h_spike):

    T = lstm_log_prob.shape[1]
    new_lattice = []
    tot_posterior_prob = 0
    h_spike = np.log(h_spike)

    time_frames = []

    for t in range(T):
        tot_posterior_prob = logsumexp_v2(lstm_log_prob[:39,t])

        if tot_posterior_prob > h_spike:
            new_lattice.append(lstm_log_prob[:,t])
            time_frames.append(t)

    new_lattice = np.array(new_lattice)
        
    return new_lattice.T, time_frames


'''
This is the second stage of Phone Lattice formation.
This stage deals with hnode parameter. The phone lattice is modified based on the 
hnode parameter decided.

Input:  outputs (LSTM probabilities nxT),
        variable_nodes (flag to truncate the nodes by a fixed number of allowed phones in each time frame or let it decide itself by using hnode parameter),
        hnode (parameter used in forming the phone lattice),
        top_n (this is the parameter to be set if the prior option of variable_nodes are considered)

Output: final_lattice (modified phone lattice)

NOTE: here the time frames do not change hence time frames are not included in this function

'''


def process_lattice_L2_hnode(outputs, variable_nodes, hnode, top_n=5):
    time_steps = outputs.shape[1]

    temp3 = outputs

    fl = np.argsort(temp3, axis=0)

    # variable_nodes is a flag that is used to whether consider dynamic number of nodes or go for fixed number of nodes at each time instance
    if variable_nodes == False:
        final_lattice = fl[-top_n:, :]
        final_lattice = final_lattice[::-1]
    
    else:
        # the main idea is to keep all the phones which are above a certain hnode for the searching purpose
        hnode = np.log(hnode)

        ind = binary_search(fl[:,0], outputs[:,0], hnode)
        data = fl[ind:, 0][::-1].tolist()

        temp_lattice = pd.DataFrame(data, columns=[0])

        
        for t in np.arange(1, time_steps):
            ind = binary_search(fl[:,t], outputs[:,t], hnode)
            data = fl[ind:, t][::-1].tolist()
            df = pd.DataFrame(data, columns=[t])
            temp_lattice = pd.concat([temp_lattice, df], axis=1, join='outer')
        
        final_lattice = temp_lattice

    return final_lattice

'''
This function collapses the time frames which have same most significant phone.

Input:  phone_lattice (phone lattice process from 2 levels)
        time_frame (list of time frames retained)

Output: new_lattice (phone lattice formed after the process)
        n_time_frames (newly retained time frames)

NOTE: This function is used when the number of phones each frame are fixed i.e. when the 
variable flag is OFF.

'''


def process_lattice_L3(phone_lattice, time_frame, blank_id=40):

    time_stamps = len(phone_lattice[0])
    n_time_frames = []

    phone_lattice = np.array(phone_lattice)

    new_lattice = [phone_lattice[:,0]]
    n_time_frames.append(time_frame[0])

    for t in np.arange(1, time_stamps):
        if new_lattice[-1][0] != phone_lattice[0,t]:
            new_lattice.append(phone_lattice[:,t])
            n_time_frames.append(time_frame[t])

    new_lattice = np.array(new_lattice)
    
    return new_lattice.T, n_time_frames

'''
This function focus on processing the lattice level 2 for variable lattice
here also check whether entire column is null or not, if yes then drop that time frame
this happens due to the chosen value of hnode.
Those frames are dropped which have same significant phone i.e. most probable phones
but also have more total non-blank probability in that time frame. This will help us
in increasing the probability occurence of the sequence inclusing that phone and if other
extra phones are added in the lattice then the chances of other low probability but desirable
phones increases.

lstm_output: LSTM Log Probabilities: nxT

Rest all the input and output variables have the same meaning as discussed previously.

'''


def process_lattice_L3_variable(lattice, time_frame, lstm_output):
    
    phones_row = len(lattice)
    time_col = len(lattice.columns)
    n_time_frames = []

    new_lattice = [np.array(lattice[:][0])]
    n_time_frames.append(time_frame[0])

    for t in np.arange(1, time_col):
        if not lattice[t][:].isnull().all():
            if new_lattice[-1][0] != lattice[t][0]:
                new_lattice.append(lattice[t][:])
                n_time_frames.append(time_frame[t])
            else:
                prev_hspike = logsumexp_v2(lstm_output[:39, n_time_frames[-1]])
                new_hspike = logsumexp_v2(lstm_output[:39, time_frame[t]])


                if new_hspike > prev_hspike:
                    new_lattice.pop()
                    new_lattice.append(lattice[t][:])
                    n_time_frames.pop()
                    n_time_frames.append(time_frame[t])    

    new_lattice = np.array(new_lattice)

    return new_lattice.T, n_time_frames
