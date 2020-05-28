from utils import *
from edit_distance import *
import math
import time

# def phone_lattice_scanning(phone_lattice, lstm_prob, keyword, blank_id=40):
#     time_stamps = phone_lattice.shape[1]

#     # recorded sequences are the complete sequences along with probabilities, edit distance from the keyword, their normalised cost and next phone
#     recorded_sequences = make_seq_arr()
#     recorded_deletion = []

#     beam = [(tuple(), (NEG_INF, INF, INF, 0.0))]

#     # ps = first phone
#     ps = keyword[0]

#     '''
#     NOTE:
#     We will consider a candidate sequence c complete if we have encountered all the possible phones in the keyword and the most probable
#     next phone is BLANK phone (=40 here). A candidate sequence is incomplete unless it has not encountered all the possible phones of the
#     keyword and an incomplete sequence has to be deleted if at the coming time stamp, we neither have same phone nor the next possible phone (pc)
#     and the BLANK phone is the only available phone with us to not to break the sequnece,
#     because we want substring and not subsequence (i.e. the derived phone sequence should be contiguous wrt time and that too with no possible blank phones)
#     '''

#     for t in range(time_stamps):
#         incomplete_sequences = make_seq_arr()
#         print(len(beam))
#         for prefix, (tot_prob, ed, cost, pc) in beam:
#             for n in phone_lattice[:,t]:

#                 phone_prob = lstm_prob[n,t]
#                 last_phone = prefix[-1] if prefix else None
#                 if n == last_phone or n == blank_id:
#                     n_prefix = prefix if prefix else (n,)
#                     n_tot_prob = prob_func(tot_prob, phone_prob)
#                     incomplete_sequences[n_prefix] = (n_tot_prob, ed, cost, pc)

#                 elif n == keyword[pc]:
#                     n_prefix = prefix + (n,)
#                     n_tot_prob = prob_func(tot_prob, phone_prob)
#                     n_pc = pc + 1

#                     if n_pc == len(keyword):
#                         recorded_sequences[n_prefix] = (n_tot_prob, ed, cost, -1)
#                     else:
#                         incomplete_sequences[n_prefix] = (n_tot_prob, ed, cost, n_pc)

#                 # adding new sequence for each instance of ps in the lattice, where ps is the first phone of the keyword
#                 if pc != 0 and n == ps:
#                     n_prefix = (n,)
#                     n_tot_prob = phone_prob
#                     # apart from the first phone, all other phones in keyword have to be added
#                     n_edit_dist = len(keyword) - 1
#                     n_cost = cost_func(n_tot_prob, n_edit_dist)
#                     n_pc = 1
#                     incomplete_sequences[n_prefix] = (n_tot_prob, n_edit_dist, n_cost, n_pc)


#         if bool(incomplete_sequences.items()):
#             beam = incomplete_sequences.items()
#         else:
#             beam = [(tuple(),(NEG_INF, INF, INF, 0))]

#     return recorded_sequences, incomplete_sequences


def phone_lattice_scanning_variable(phone_lattice, lstm_prob, keyword, blank_id=40):
    '''
    This function is for the variable nodes in the lattice and not fixed.
    We will implement this using the pandas library
    '''

    time_stamps = phone_lattice.shape[1]

    recorded_sequences = make_seq_arr()

    beam = [(tuple(), (NEG_INF, INF, INF, 0))]

    ps = keyword[0]

    for t in range(time_stamps):
        incomplete_sequences = make_seq_arr()
        for prefix, (tot_prob, ed, cost, pc) in beam:
            for n in phone_lattice[:,t]:

                if np.isnan(n):
                    continue
                n = int(n)
                phone_prob = lstm_prob[n,t]

                last_phone = prefix[-1] if prefix else None
                if n == last_phone or n == blank_id:
                    n_prefix = prefix if prefix else (n,)
                    n_tot_prob = prob_func(tot_prob, phone_prob)
                    incomplete_sequences[n_prefix] = (n_tot_prob, ed, cost, pc)

                elif n == keyword[pc]:
                    n_prefix = prefix + (n,)
                    n_tot_prob = prob_func(tot_prob, phone_prob)
                    n_pc = pc + 1

                    if n_pc == len(keyword):
                        recorded_sequences[n_prefix] = (n_tot_prob, ed, cost, -1)
                    else:
                        incomplete_sequences[n_prefix] = (n_tot_prob, ed, cost, n_pc)

                if pc != 0 and n == ps:
                    n_prefix = (n,)
                    n_tot_prob = phone_prob
                    n_edit_dist = len(keyword) - 1
                    n_cost = cost_func(n_tot_prob, n_edit_dist)
                    n_pc = 1
                    incomplete_sequences[n_prefix] = (n_tot_prob, n_edit_dist, n_cost, n_pc)

        if bool(incomplete_sequences.items()):
            beam = incomplete_sequences.items()#, key= lambda x:prob_func(*x[1]), reverse=True)
        else:
            beam = [(tuple(),(NEG_INF, INF, INF, 0))]

    return recorded_sequences, incomplete_sequences


def pls_v3(phone_lattice, lstm_prob, keyword, ids_prob, blank_id=40):
    '''
    This function is for finding the sequence along with implementing the insertion, deletion and subsitutuion probabilities
    '''

    insert_prob, delete_prob, substitute_prob = pickle.load(ids_pickle)
    # insert_prob = ids_prob[0]
    # delete_prob = ids_prob[1]
    # substitute_prob = ids_prob[2]

    time_stamps = phone_lattice.shape[1]

    # print(time_stamps)

    recorded_sequences = make_seq_arr()
    incorrect_sequences = make_seq_arr()

    beam = [(tuple(), (NEG_INF, INF, INF, 0))]

    ps = keyword[0]

    for t in range(time_stamps):
        incomplete_sequences = make_seq_arr()
        for prefix, (tot_prob, ed, cost, pc) in beam:
            for n in phone_lattice[:,t]:
                # print(n, end=" ")
                if np.isnan(n):
                    continue

                n = int(n)
                phone_prob = lstm_prob[n,t]
                # print(phone_prob)
                last_phone = prefix[-1] if prefix else None

                if n == last_phone or n == blank_id:
                    n_prefix = prefix if prefix else (n,)
                    n_tot_prob = prob_func(tot_prob, phone_prob)
                    incomplete_sequences[n_prefix] = (n_tot_prob, ed, cost, pc)

                elif n == keyword[pc]:
                    n_prefix = prefix + (n,)
                    n_tot_prob = prob_func(tot_prob, phone_prob)
                    n_pc = pc + 1

                    if n_pc == len(keyword):
                        recorded_sequences[n_prefix] = (n_tot_prob, ed, cost, -1)
                    else:
                        incomplete_sequences[n_prefix] = (n_tot_prob, ed, cost, n_pc)

                else:
                    n_prefix = prefix + (n,)
                    n_tot_prob = prob_func(tot_prob, phone_prob)


                if pc != 0 and n == ps:
                    n_prefix = (n,)
                    n_tot_prob = phone_prob
                    n_edit_dist = len(keyword) - 1
                    n_cost = cost_func(n_tot_prob, n_edit_dist)
                    n_pc = 1
                    incomplete_sequences[n_prefix] = (n_tot_prob, n_edit_dist, n_cost, n_pc)

        if bool(incomplete_sequences.items()):
            beam = incomplete_sequences.items()#, key= lambda x:prob_func(*x[1]), reverse=True)
        else:
            beam = [(tuple(),(NEG_INF, INF, INF, 0))]
        # print(t, len(beam), len(recorded_sequences.items()))
        # print(incomplete_sequences.items())

    return recorded_sequences, beam

def dynamic_pls_v4(lstm_prob, phone_lattice, time_frames, keyword, blank_id, ids_prob, arg):

    # print("Following results are rom pls_v4:")

    time_steps = len(time_frames)

    # recorded_sequences = make_seq_arr()

    beam = [(tuple(), (NEG_INF, INF, INF, 0.0))]

    for t in range(time_steps):

        # print("---------- " + str(t) + " ----------")

        incomplete_sequences = make_seq_arr()

        for prefix, (tot_prob, ed, cost, pc) in beam:

            # print("Prefix: " + str(prefix))

            for n in phone_lattice[:, t]:

                if np.isnan(n):
                    continue

                n = int(n)

                phone_prob = lstm_prob[n,t]

                last_phone = prefix[-1] if prefix else None

                if n == last_phone or n == blank_id:
                    if prefix:
                        # n_prefix = prefix
                        n_prob = prob_func(prob, phone_prob)
                        incomplete_sequences[prefix] = (n_prob, ed, cost, dp)

                elif n != 39:
                    n_prefix = prefix + (n,)
                    n_tot_prob = prob_func(tot_prob, phone_prob)

                    n_ed = edit_distance(ids_prob, n_prefix, keyword)
                    # print("Adding new phone: " + str(n_prefix) + " ED:" + str(n_ed))
                    n_cost = cost_func(n_tot_prob, n_ed, arg)
                    incomplete_sequences[n_prefix] = (n_tot_prob, n_ed, n_cost, pc)

                if n != 39 and n != blank_id:
                    n_prefix = (n,)
                    n_tot_prob = phone_prob
                    # print("Initializing: " + str(n_prefix))
                    n_edit_dist = edit_distance(ids_prob, n_prefix, keyword)
                    n_cost = cost_func(n_tot_prob, n_edit_dist, arg)
                    n_pc = 1
                    incomplete_sequences[n_prefix] = (n_tot_prob, n_edit_dist, n_cost, n_pc)

        if bool(incomplete_sequences.items()):
            beam = sorted(incomplete_sequences.items(),key=lambda a:a[1][2])[:30]
        else:
            beam = [(tuple(),(NEG_INF, INF, INF, 0))]
        # print(t, len(beam), len(recorded_sequences.items()))
        # print(incomplete_sequences.items())

    return incomplete_sequences


'''
Parameters:
lstm_log_Prob: LSTM Log probabilities
final_Lattice: final Phone lattice
Keyword: Target sequence
Time_Frames: the retained time frames in the final lattice
IDS_prob: insert, delete and substitute probabilities
Arg: argument to choose cost function (int type)
Smax: threshold below which the sequence will be noted
V: parameter for pruning
'''

def Modified_Dynamic_PLS(lstm_log_Prob, final_Lattice, Keyword, Time_Frames, IDS_prob, Arg, V):

    final_Lattice = final_Lattice.values

    blank_id = 40

    min_len, max_len = compute_len_Range(Keyword, IDS_prob, 2)
    # print(min_len, max_len)
    # time.sleep(10)

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

        print("---------- " + str(t) + " ----------")
        incomplete_sequences = make_seq_arr(N)

        for n in final_Lattice[:, t]:

            if np.isnan(n):
                continue

            n = int(n)

            for prefix, (prob, cost, tf, dp) in beam:

                # print("Prefix: " + str(prefix))

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
                    # if n_len >= min_len and n <= max_len:
                        # print("Comparing the prefix for ED: " + str(n_prefix))
                    n_dp = edit_distance_v2(n, Keyword, dp, IDS_prob)
                    # else:
                        # n_dp = dp
                    n_ed = n_dp[N]
                    # if n_len >= min_len:
                    n_cost = cost_func(n_prob, n_ed, Arg)
                    # else:
                        # n_cost = cost
                    print("Adding new phone: " + str(n_prefix) + "     ED:" + str(n_ed))

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
                print("Initializing: " + str(n_prefix))

        if bool(incomplete_sequences.items()):
            beam = sorted(incomplete_sequences.items(), key=lambda a:a[1][0])[:V]
            print("Beam Length for next iteration: " + str(len(beam)))
        else:
            beam = [(tuple(),(NEG_INF, INF, 0, np.zeros(N)))]

    return recorded_sequences


# Below all are fixed scanning methods


def Fixed_PLS_single_pronunciation(phone_lattice, time_frame, lstm_prob, keyword, blank_id=40):

    time_stamps = phone_lattice.shape[1]

    # print(time_stamps)

    recorded_sequences = make_seq_arr_fixed()
    # recorded_deletion = []

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

    return recorded_sequences, incomplete_sequences


# following function is used to calculate the max and min phone probabilities in the sequence detected

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

    return recorded_sequences, incomplete_sequences
