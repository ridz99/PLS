import json
import pandas as pd
import pickle
import numpy as np
from PLS.process_lattice import *
from PLS.phone_lattice_scanning import *
from PLS.utils import *

# Just read the pickle file for the final lattice
def read_pickle_lattice_file(filename):

    list_lattice_time_frames = []

    fptr = open(filename, "rb")
    data = pickle.load(fptr)

    list_lattice = []
    list_time_frames = []

    N = len(data)

    for i in range(N):
        df = pd.DataFrame(data[i][0])
        list_time_frames.append(data[i][1])
        list_lattice.append(df)

    return list_lattice, list_time_frames


'''
Find which frames are missing from the phone lattice formed from the LSTM output
'''
def calculate_id(tot_time_steps, final_lattice, time_frame, grd_truth):

    new_phones = np.zeros(39)
    deleted_phones = np.zeros(39)

    tot_time_frames_fl = time_frame.shape[0]
    len_phone_seq =  len(grd_truth)

    avg_frames_per_phone = float(tot_time_steps)/float(len_phone_seq)

    window_size = 1

    num_phones_each_tf = np.count_nonzero(~np.isnan(final_lattice), axis=0)
    seq_pos = 0
    time_pos = 0

    while seq_pos < len_phone_seq and time_pos < tot_time_frames_fl:
        tf = time_frame[time_pos]

        if (time_pos + 1) < tot_time_frames_fl:
            window_size = math.floor(float(time_frame[time_pos+1] - time_frame[time_pos])/avg_frames_per_phone)
        else:
            window_size = math.floor(float(time_frame[tot_time_frames_fl-1] - time_frame[time_pos])/avg_frames_per_phone)

        if (seq_pos + window_size) < len_phone_seq:
            cur_phones = grd_truth[seq_pos:(seq_pos + window_size+1)]
        else:
            cur_phones = grd_truth[seq_pos:]

        match_arr = np.array([(phone == final_lattice[:, time_pos]).any() for phone in cur_phones])
        if match_arr.any():
            if not match_arr[0]:
                deleted_phones[grd_truth[seq_pos]] += 1

            # find the first true value of the window array
            res = next((i for i, j in enumerate(match_arr) if j), None)
            seq_pos += (res + 1)
        else:
            '''
            it is compulsory that neither of the time frame should have complete NaN in the final lattice
            '''
            if num_phones_each_tf[time_pos] > 0 and not (int(final_lattice[0, time_pos]) == 40):
                new_phones[int(final_lattice[0, time_pos])] += 1

            deleted_phones[grd_truth[seq_pos]] += 1
            seq_pos += 1

        time_pos += 1

    return new_phones, deleted_phones


'''
This is for single audio file testing.

Input: 	LSTM_output (nxT where T is the time frames)
		IDS_prob (insertion, deletion, substitution probabilities)
		hspike, hnode, V (hyperparameters)
		PLS_Arg, Cost_Arg (int type chosing parameters)

Output:	list of all possible matches along with start and end time frame, and other related data

'''

def test_one_utterance(LSTM_output, IDS_prob, hspike, hnode, keyword, PLS_Arg, Cost_Arg, V):

	LSTM_output = LSTM_output.T

	log_prob = np.log(LSTM_output)

	phone_lattice_raw, time_frame = process_lattice_hspike(log_prob, hspike)

	final_lattice = process_lattice_L2_hnode(phone_lattice_raw, variable, hnode)

	if variable:
		final_lattice, time_frame = process_lattice_L3_variable(final_lattice, time_frame, log_prob)
	else:
		final_lattice, time_frame = process_lattice_L3(final_lattice, time_frame)

	num_hits = 0

	if PLS_Arg == 1:
		recorded_seq = Fixed_PLS_single_pronunciation(phone_lattice, time_frames, LSTM_prob, keyword[0])
	elif PLS_Arg == 2:
		recorded_seq = Fixed_PLS_multi_pronunciation(phone_lattice, time_frames, LSTM_prob, keyword)
	else:
		recorded_seq = Modified_Dynamic_PLS(LSTM_prob, phone_lattice, keyword[0], time_frames, ids_prob, cost_Arg, V)

	if bool(recorded_seq):
		num_tp += 1
		ans = sorted(recorded_seq.items(), key = lambda a: a[1][0], reverse=True)

	return ans


def test_multiple_utterance(LSTM_output, IDS_prob, keyword, hspike, hnode, PLS_Arg, Cost_Arg, V):

	n_test = len(LSTM_output)

	answers = []

	for i in range(n_test):

		ans = test_one_utterance(LSTM_output[i].T, IDS_prob, keyword, hspike, hnode, PLS_Arg, Cost_Arg, V)

		answers.append(ans)

	return answers
