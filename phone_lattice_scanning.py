# DONE

from PLS.utils import *
from PLS.edit_distance import *
import math
import time


'''
Modified Dynamic PLS method takes into account the insertion, deletion and substitution
probabilities obtained during the training model. These probabilities are incorporated
into the dynamic programming algorithm in terms of edit distance and later the cummulative
calculation of the occurence probability of the sequence detected and its edit distance
from the keyword are taken into consideration for determining the most probable detection.

Input:  lstm_log_Prob (LSTM Log probabilities)
        final_Lattice (Phone lattice generated)
        Keyword (Target sequence)
        Time_Frames (the retained time frames in the phone lattice)
        IDS_prob (insert, delete and substitute probabilities)
        Arg (argument to choose cost function from the list mentioned in utils.py)
        V (parameter for pruning)

Output: recorded_sequences (list of all detected sequences)

NOTE:   lstm_log_Prob: (nxT)
        final_lattice: preformed
        Keyword: 1D list
        Time_Frames: preformed
        IDS_prob: preformed (list[0] contains insertion, list[1] contains deletion, list[2] is a matrix of substitution probabilities)
        Arg: int
        V: int
        recorded_sequences: list of tuples with key being the keyword detected, and the value of each key is a list of occurence probability, eit distance, cost, start, end time frame
'''

def Modified_Dynamic_PLS(lstm_log_Prob, final_Lattice, Keyword, Time_Frames, IDS_prob, Arg, V):

    # final_Lattice = final_Lattice.values

    blank_id = 40

    min_len, max_len = compute_len_Range(Keyword, IDS_prob, 2)

    time_steps = len(Time_Frames)

    N = len(Keyword)

    recorded_sequences = make_record_seq()

    # calculate the first column of ED
    insert_prob = IDS_prob[0]
    insert_prob = np.array(logarithm_mat(insert_prob))
    phones_keyword = [int(i) for i in Keyword]
    first_column_ED = np.zeros(N+1)
    for i in np.arange(1, N):
        # first_column_ED[i] = prob_func(first_column_ED[i-1], insert_prob[int(Keyword[i-1])])
        first_column_ED[i] = first_column_ED[i-1] + insert_prob[int(Keyword[i-1])]
    #end

    beam = [(tuple(), (NEG_INF, INF, 0, first_column_ED))]

    for t in range(time_steps):

        incomplete_sequences = make_seq_arr(N)

        for n in final_Lattice[:, t]:

            if np.isnan(n):
                continue

            n = int(n)

            for prefix, (prob, cost, tf, dp) in beam:

                phone_prob = lstm_log_Prob[n,t]

                last_phone = prefix[-1] if prefix else None

                if n == last_phone or n == blank_id:
                    if prefix:
                        n_prob = prob_func(prob, phone_prob)
                        incomplete_sequences[prefix] = (n_prob, cost, tf, dp)

                elif n != 39:
                    n_prefix = prefix + (n,)
                    n_prob = prob_func(prob, phone_prob)
                    n_len = len(n_prefix)
                    n_dp = edit_distance_v2(n, Keyword, dp, IDS_prob)
                    n_ed = n_dp[N]
                    n_cost = cost_func(n_prob, n_ed, Arg)

                    if n_len >= min_len and n_len <= max_len: # and n_cost < Smax
                        recorded_sequences[n_prefix] = (n_prob, n_ed, n_cost, tf, Time_Frames[t])

                    if n_len <= max_len:
                        incomplete_sequences[n_prefix] = (n_prob, n_cost, tf, n_dp)

            if n != 39 and n != blank_id:
                n_prefix = (n,)
                n_prob = phone_prob
                n_dp = edit_distance_v2(n, Keyword, first_column_ED, IDS_prob)
                n_ed = n_dp[N]
                n_cost = cost_func(n_prob, n_ed, Arg)
                incomplete_sequences[n_prefix] = (n_prob, n_cost, Time_Frames[t], n_dp)

        if bool(incomplete_sequences.items()):
            beam = sorted(incomplete_sequences.items(), key=lambda a:a[1][0])[:V]
        else:
            beam = [(tuple(),(NEG_INF, INF, 0, np.zeros(N)))]

    return recorded_sequences

'''
This method is the exact match phone lattice scanning method for single pronunciation.
The exact phone sequence of the keyword will be traced in the phone lattice formed.

Input:  phone_lattice (phone lattice generated)
        time_frame (retained time frames out of all the time frames considered in the LSTM model)
        lstm_prob (LSTM log probabilities)
        keyword (Target phone sequence)

Output: recorded_sequences (list of all detected sequences)

NOTE:   phone_lattice: preformed
        time_frame: preformed
        lstm_prob: nxT
        keyword: 1D list of phone sequence
        recorded_sequences: list of tuples where the key represents the matched/detected sequence and its value consists of list of occurence probability, start time frame and pc value (this value is for algorithm and not for practical purpose)

'''

def Fixed_PLS_single_pronunciation(phone_lattice, time_frame, lstm_prob, keyword, blank_id=40):

    time_stamps = phone_lattice.shape[1]

    recorded_sequences = make_seq_arr_fixed()

    beam = [(tuple(), (NEG_INF, -1, 0))]

    ps = keyword[0]

    for t in range(time_stamps):
        incomplete_sequences = make_seq_arr_fixed()
        for prefix, (tot_prob, tf, pc) in beam:
            for n in phone_lattice[:,t]:

                if np.isnan(n):
                    continue

                n = int(n)
                phone_prob = lstm_prob[n,t]

                last_phone = prefix[-1] if prefix else None
                if n == last_phone or n == blank_id:
                    incomplete_sequences[prefix] = (tot_prob, tf, pc)

                elif n == keyword[pc]:
                    n_prefix = prefix + (n,)
                    n_tot_prob = prob_func(tot_prob, phone_prob)
                    n_pc = pc + 1

                    if n_pc == len(keyword):
                        recorded_sequences[n_prefix] = (n_tot_prob, tf, -1)
                    else:
                        incomplete_sequences[n_prefix] = (n_tot_prob, tf, n_pc)

                if pc != 0 and n == ps:
                    n_prefix = (n,)
                    n_tot_prob = phone_prob
                    n_tf = time_frame[t]
                    n_pc = 1
                    incomplete_sequences[n_prefix] = (n_tot_prob, n_tf, n_pc)

        if bool(incomplete_sequences.items()):
            beam = incomplete_sequences.items()#, key= lambda x:prob_func(*x[1]), reverse=True)
        else:
            beam = [(tuple(),(NEG_INF, -1, 0))]

    return recorded_sequences#, incomplete_sequences

'''
This method focuses on exact match among various pronunciations of a single keyword.
If either of the pronunciation matches exactly with the detected sequence then it is 
considered as a hit for that corresponding keyword.

Input:  phone_lattice (phone lattice formed)
        time_frame (time frames retained after phone lattice formation)
        lstm_prob (LSTM log probabilities)
        keywords (target phone sequences) 

Output: recorded_sequences (list of all possible detected sequences)

NOTE: all the data type of the variables are same as discussed in previous methods.

'''


def Fixed_PLS_multi_pronunciations(phone_lattice, time_frame, lstm_prob, keywords):

    blank_id = 40

    num_keyword = len(keywords)

    len_keywords = [len(keywords[i]) for i in range(num_keyword)]

    time_steps = phone_lattice.shape[1]

    recorded_sequences = make_seq_arr_fixed_key(num_keyword)

    beam = [(tuple(), (NEG_INF, -1, [0]*num_keyword))]

    ps = [keywords[i][0] for i in range(num_keyword)]


    for t in range(time_steps):

        incomplete_sequences = make_seq_arr_fixed_key(num_keyword)


        for prefix, (prob, tf, pc) in beam:

            for n in phone_lattice[:,t]:

                if np.isnan(n):
                    continue

                n = int(n)

                phone_prob = lstm_prob[n,t]

                last_phone = prefix[-1] if prefix else None

                pc_existence = np.empty(num_keyword, dtype=bool)
                
                for i in range(num_keyword):
                    
                    k_pc = pc[i]
                    pc_existence[i] = (n == keywords[i][k_pc])

                if n == last_phone or n == blank_id:
                    incomplete_sequences[prefix] = (prob, tf, pc)

                elif pc_existence.any():
                    n_prefix = prefix + (n,)
                    n_prob = prob_func(phone_prob, prob)

                    n_pc = []

                    for i in range(num_keyword):
                        if pc_existence[i]:
                            n_pc.append(pc[i]+1)
                        else:
                            n_pc.append(-1)

                    finish_seq = np.empty(num_keyword, dtype=bool)

                    for i in range(num_keyword):
                        finish_seq[i] = (len_keywords[i] == n_pc[i])

                    if finish_seq.any():
                        recorded_sequences[n_prefix] = (n_prob, tf, n_pc)
                        
                    else:
                        incomplete_sequences[n_prefix] = (n_prob, tf, n_pc)

                find_truth = (np.array(ps) == n)
                if (np.array(pc) != 0).any() and find_truth.any():
                    n_prefix = (n,)
                    n_prob = phone_prob
                    n_tf = time_frame[t]

                    n_pc = []
                    for i in range(num_keyword):
                        if find_truth[i]:
                            n_pc.append(1)
                        else:
                            n_pc.append(0)

                    incomplete_sequences[n_prefix] = (n_prob, n_tf, n_pc)

        if bool(incomplete_sequences.items()):
            beam = incomplete_sequences.items()
        else:
            beam = [(tuple(), (NEG_INF, -1, [0]*num_keyword))]

    return recorded_sequences#, incomplete_sequences


'''
All the input and output variables of this method holds the same reference as they earlier 
did for the prior method. Just in this function, we calculate one more factor of the 
sequence i.e. the maximun and minimum phone probability among the detected sequence.
This was one of the observation function made for short keywords poor performance.
'''

def Fixed_PLS_multi_pronunciations_max_min(phone_lattice, time_frame, lstm_prob, keywords):

    phone_lattice = phone_lattice.values

    blank_id = 40

    num_keyword = len(keywords)
    
    len_keywords = [len(keywords[i]) for i in range(num_keyword)]

    time_steps = phone_lattice.shape[1]

    recorded_sequences = make_seq_arr_fixed_key_v2(num_keyword)

    beam = [(tuple(), (NEG_INF, -1, [0]*num_keyword, NEG_INF, 0))]

    ps = [keywords[i][0] for i in range(num_keyword)]


    for t in range(time_steps):

        incomplete_sequences = make_seq_arr_fixed_key_v2(num_keyword)


        for prefix, (prob, tf, pc, max_p, min_p) in beam:

            for n in phone_lattice[:,t]:

                if np.isnan(n):
                    continue

                n = int(n)

                
                phone_prob = lstm_prob[n,t]

                last_phone = prefix[-1] if prefix else None

                pc_existence = np.empty(num_keyword, dtype=bool)
                
                for i in range(num_keyword):
                
                    k_pc = pc[i]
                    pc_existence[i] = (n == keywords[i][k_pc])

                if n == last_phone or n == blank_id:
                    incomplete_sequences[prefix] = (prob, tf, pc, max_p, min_p)

                elif pc_existence.any():
                    n_prefix = prefix + (n,)
                    n_prob = prob_func(phone_prob, prob)

                    n_max_p = max(max_p, phone_prob)
                    n_min_p = min(min_p, phone_prob)

                    n_pc = []

                    for i in range(num_keyword):
                        if pc_existence[i]:
                            n_pc.append(pc[i]+1)
                        else:
                            n_pc.append(-1)

                    finish_seq = np.empty(num_keyword, dtype=bool)

                    for i in range(num_keyword):
                        finish_seq[i] = (len_keywords[i] == n_pc[i])

                    if finish_seq.any():
                        recorded_sequences[n_prefix] = (n_prob, tf, n_pc, n_max_p, n_min_p)
                
                    else:
                        incomplete_sequences[n_prefix] = (n_prob, tf, n_pc, n_max_p, n_min_p)

                find_truth = (np.array(ps) == n)
                if (np.array(pc) != 0).any() and find_truth.any():
                    n_prefix = (n,)
                    n_prob = phone_prob
                    n_tf = time_frame[t]

                    n_max_p = max(max_p, phone_prob)
                    n_min_p = min(min_p, phone_prob)

                    n_pc = []
                    for i in range(num_keyword):
                        if find_truth[i]:
                            n_pc.append(1)
                        else:
                            n_pc.append(0)

                    incomplete_sequences[n_prefix] = (n_prob, n_tf, n_pc, n_max_p, n_min_p)

        if bool(incomplete_sequences.items()):
            beam = incomplete_sequences.items()
        else:
            beam = [(tuple(), (NEG_INF, -1, [0]*num_keyword, NEG_INF, 0))]

    return recorded_sequences#, incomplete_sequences
