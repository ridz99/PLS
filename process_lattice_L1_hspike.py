import numpy as np
from utils import *

def process_lattice_hspike(lstm_log_prob, h_spike):

    # print("process_lattice_hspike:", end=" ")
    T = lstm_log_prob.shape[1]
    # print(T)
    new_lattice = []
    tot_posterior_prob = 0
    h_spike = np.log(h_spike)
    # print(h_spike)
    time_frames = []

    for t in range(T):
        tot_posterior_prob = logsumexp_v2(lstm_log_prob[:39,t])
        # print(math.exp(tot_posterior_prob))
        # print(tot_posterior_prob)
        if tot_posterior_prob > h_spike:
            new_lattice.append(lstm_log_prob[:,t])
            time_frames.append(t)

    new_lattice = np.array(new_lattice)
    
    # print(new_lattice.T.shape)
    
    return new_lattice.T, time_frames