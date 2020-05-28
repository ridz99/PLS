import numpy as np
from utils import logsumexp_v2

def process_lattice(phone_lattice, time_frame, blank_id=40):

    '''
    This function collapses the time frames which have same most significant phone
    '''

    # print("process_lattice:", end=" ")

    time_stamps = len(phone_lattice[0])
    n_time_frames = []
    # print("process_lattice:")
    # print("time_stamps: " + str(time_stamps))    

    phone_lattice = np.array(phone_lattice)

    new_lattice = [phone_lattice[:,0]]
    n_time_frames.append(time_frame[0])
    # n_lstm_prob = []
    # print(new_lattice)

    for t in np.arange(1, time_stamps):
        # print(new_lattice[-1][0], phone_lattice[0,t])
        if new_lattice[-1][0] != phone_lattice[0,t]:
            new_lattice.append(phone_lattice[:,t])
            n_time_frames.append(time_frame[t])
            # print(t)

    new_lattice = np.array(new_lattice)
    
    # print(new_lattice.T.shape)

    return new_lattice.T, n_time_frames

def process_lattice_v3(lattice, time_frame, lstm_output):
    '''
    This function focus on processing the lattice level 2 for variable lattice
    here also check whether entire column is null or not, if yes then drop that time frame
    this happens due to the chosen value of hnode
    '''
    
    phones_row = len(lattice)
    time_col = len(lattice.columns)
    n_time_frames = []

    # print("process_lattice_v3:")
    # print("phones_row: " + str(phones_row) + " time_col: " + str(time_col))
    # print("length of time_frame: " + str(len(time_frame)))

    # print(np.array(lattice[:][0]))

    new_lattice = [np.array(lattice[:][0])]
    # print(new_lattice)
    n_time_frames.append(time_frame[0])

    for t in np.arange(1, time_col):
        # print(new_lattice[-1][0], lattice[t][0])
        # if new_lattice[-1][0] != lattice[t][0] and (not lattice[t][:].isnull().all()):
        #     new_lattice.append(lattice[t][:])
        #     n_time_frames.append(time_frame[t])
        if not lattice[t][:].isnull().all():
            if new_lattice[-1][0] != lattice[t][0]:
                new_lattice.append(lattice[t][:])
                n_time_frames.append(time_frame[t])
            else:
                prev_hspike = logsumexp_v2(lstm_output[:39, n_time_frames[-1]])
                new_hspike = logsumexp_v2(lstm_output[:39, time_frame[t]])

                # print(n_time_frames[-1], time_frame[t], prev_hspike, new_hspike)

                if new_hspike > prev_hspike:
                    new_lattice.pop()
                    new_lattice.append(lattice[t][:])
                    n_time_frames.pop()
                    n_time_frames.append(time_frame[t])    

    new_lattice = np.array(new_lattice)

    return new_lattice.T, n_time_frames
