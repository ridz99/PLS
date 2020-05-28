import numpy as np
import pandas as pd
import pickle
from process_lattice_L1_hspike import *
from generate_lattice import *
from process_lattice_L2_hnode import *
from phone_lattice_scanning import *
from softmax import *

'''
Find which frames are missing from the phone lattice formed from the LSTM output
'''
def calculate_id(tot_time_steps, final_lattice, time_frame, grd_truth):

    new_phones = np.zeros(39)
    deleted_phones = np.zeros(39)

    tot_time_frames_fl = time_frame.shape[0]
    # tot_time_steps = lstm_lattice.shape[1]
    len_phone_seq =  len(grd_truth)

    avg_frames_per_phone = float(tot_time_steps)/float(len_phone_seq)

    window_size = 1

    num_phones_each_tf = np.count_nonzero(~np.isnan(final_lattice), axis=0)
    # print(num_phones_each_tf)
    seq_pos = 0
    time_pos = 0

    # print(tot_time_frames_fl)

    while seq_pos < len_phone_seq and time_pos < tot_time_frames_fl:
        # curr_phone = grd_truth[seq_pos]
        tf = time_frame[time_pos]

        # print("time_pos " + str(time_pos))

        if (time_pos + 1) < tot_time_frames_fl:
            window_size = math.floor(float(time_frame[time_pos+1] - time_frame[time_pos])/avg_frames_per_phone)
        else:
            window_size = math.floor(float(time_frame[tot_time_frames_fl-1] - time_frame[time_pos])/avg_frames_per_phone)

        # print(window_size)

        if (seq_pos + window_size) < len_phone_seq:
            cur_phones = grd_truth[seq_pos:(seq_pos + window_size+1)]
        else:
            cur_phones = grd_truth[seq_pos:]

        # print("cur_phones:", end=" ")
        # print(cur_phones)

        # print(final_lattice[:,time_pos])
        match_arr = np.array([(phone == final_lattice[:, time_pos]).any() for phone in cur_phones])
        # print(match_arr)
        if match_arr.any():
            if not match_arr[0]:
                deleted_phones[grd_truth[seq_pos]] += 1

            # find the first true value of the window array
            res = next((i for i, j in enumerate(match_arr) if j), None)
            # print("res:"+str(res))
            seq_pos += (res + 1)
        else:
            '''
            it is compulsory that neither of the time frame should have complete NaN in the final lattice
            '''
            if num_phones_each_tf[time_pos] > 0 and not (int(final_lattice[0, time_pos]) == 40):
                # print(seq_pos, time_pos)
                new_phones[int(final_lattice[0, time_pos])] += 1

            deleted_phones[grd_truth[seq_pos]] += 1
            seq_pos += 1

        time_pos += 1

    return new_phones, deleted_phones


def testing(lstm_output, hspike, variable, grd_truth, hnode=0.005):

    log_prob = softmax(lstm_output)

    phone_lattice_raw, time_frame = process_lattice_hspike(log_prob, hspike)

    final_lattice = generate_lattice(phone_lattice_raw, variable, hnode)

    time_frame_1 = np.array(time_frame).reshape(1,len(time_frame))

    if variable:
        final_lattice, time_frame = process_lattice_v3(final_lattice, time_frame, log_prob)
    else:
        final_lattice, time_frame = process_lattice(final_lattice, time_frame)


    time_frame = np.array(time_frame).reshape(1,len(time_frame))

    if variable:
        r_s, inc_s = phone_lattice_scanning_variable(final_lattice, output_prob, grd_truth)
    else:
        r_s, inc_s = phone_lattice_scanning(final_lattice, output_prob, grd_truth)

    return final_lattice, time_frame, bool(r_s)


'''
lstm_output: softmax of the LSTM probabilities 
'''    

def testing_v2(lstm_output, hspike, variable, keyword, hnode, ids_prob, arg, Smax, V):
    
    # if softmax is not carried out then use the softmax function and comment the log command because
    # log is already included in calculating the softmax of the lstm_output

    # log_prob = softmax(lstm_output)

    log_prob = np.log(lstm_output)

    phone_lattice_raw, time_frame = process_lattice_hspike(log_prob, hspike)

    final_lattice = generate_lattice(phone_lattice_raw, variable, hnode)

    if variable:
        final_lattice, time_frame = process_lattice_v3(final_lattice, time_frame, log_prob)
    else:
        final_lattice, time_frame = process_lattice(final_lattice, time_frame)


    if variable:
        # inc_s = pls_v4(log_prob, final_lattice, time_frame, keyword, 40, ids_prob, arg)
        r_s = pls_v5(log_prob, final_lattice, keyword, time_frame, ids_prob, arg, Smax, V)
    else:
        r_s, inc_s = phone_lattice_scanning(final_lattice, output_prob, grd_truth)

    return final_lattice, time_frame, r_s

    

def get_the_histo(sa1_prob, sa2_prob, arg, key):

    tot_test = len(sa1_prob)

    # darksuit
    # dcl d aa r kcl k s uw dx
    key_1 = [27, 7, 0, 28, 27, 19, 29, 34, 9]

    # washwater
    # w aa sh epi w aa dx er
    key_2 = [36, 0, 30, 27, 36, 0, 9, 11]

    # greasy
    # gcl g r iy s iy
    key_3 = [27, 14, 28, 17, 29, 17]

    correct_sa1 = []

    print("Going for correct sa1" + " " + str(arg))

    for i in range(tot_test):
        # print("Index: " + str(i) + " " + str(arg))
        test_ex = sa1_prob[i][0].T
        p, tf, seq = testing_v2(test_ex, 0.1, True, key, 0.005, ids_prob, arg)
        ans = sorted(seq.items(), key=lambda a: a[1][2])
        correct_sa1.append(ans[0][1][1])


    incorrect_sa2 = []

    print("Going for incorrect sa2" + " " + str(arg))

    for i in range(tot_test):
        # print("Index: " + str(i) + " " + str(arg))
        test_ex = sa2_prob[i][0].T
        p, tf, seq = testing_v2(test_ex, 0.1, True, key, 0.005, ids_prob, arg)
        ans = sorted(seq.items(), key=lambda a: a[1][2])
        incorrect_sa2.append(ans[0][1][1])

    print("Complete...." + " " + str(arg))

    return correct_sa1, incorrect_sa2


def testing_v3(lstm_output, hspike, keyword, hnode):

    log_prob = np.log(lstm_output)
    # log_prob = lstm_output
    print(log_prob.shape)

    phone_lattice_raw, time_frame = process_lattice_hspike(log_prob, hspike)

    final_lattice = generate_lattice(phone_lattice_raw, True, hnode)

    final_lattice, time_frame = process_lattice_v3(final_lattice, time_frame, log_prob)

    # r_s, inc_s = phone_lattice_scanning_variable_v3(final_lattice, time_frame, log_prob, keyword)

    return final_lattice, time_frame#, r_s, inc_s

def get_histo_v2(sa1_prob, key, hspike, hnode):

    tot_test = len(sa1_prob)

    correct_sa1_prob = []
    final_TP = []
    final_FN = []

    print("Starting the testing for the fixed keyword matching for:")
    print("hspike: " + str(hspike) + " hnode: " + str(hnode))

    for k in key:
        print("PLS keyword: " + str(k))

        num_tp = 0
        num_fn = 0
        num_fp = 0

        print("Checking for True Positive and False Negative among 168 sentences where the keyword is present in each sentence:")
        for i in range(tot_test):
            test_ex = sa1_prob[i][0].T
            p, tf, in_seq, seq  = testing_v3(test_ex, hspike, k, hnode)
            ans = sorted(seq.items(), key=lambda a: a[1][0])
            if bool(seq):
                num_tp += 1
                correct_sa1_prob.append(ans[0][1][0])
            else:
                num_fn += 1

        print("#True Positive: " + str(num_tp) + " #False Negative: " + str(num_fn))

        final_TP.append(num_tp)
        final_FN.append(num_fn)

    print("Complete....")

    return final_TP, final_FN, correct_sa1_prob


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


# lstm_output: this should be lstm output without doing softmax
def testing_all(filename, lstm_output, keywords, hs, hn):

    # n_keywords = len(keywords)

    final_lattices, time_frames = read_pickle_lattice_file(filename)

    n_test = len(final_lattices)

    hit_probs = []

    num_hits = []
    answers = []

    # for i in range(n_keywords):
        # key = keywords[i]

    with open("probs/GRU_5_384_79_probs.pkl", "rb") as f:
        IDS_prob = pickle.load(f)

    num_fp = 0
    # hit_probs.append([])
    for j in range(1):
        # j = 1116
        print("Test #" + str(j))
        log_prob = lstm_output[j][0].T
        log_prob = np.log(log_prob)
        lattice = final_lattices[j]
        # print(lattice)
        tf = time_frames[j]

        # r_S, inc_S = phone_lattice_scanning_variable_v4(lattice, tf, log_prob, keywords)
        r_S = dmpls(log_prob, lattice, keywords, tf, IDS_prob, 3, 20)

        if bool(r_S):
            ans = sorted(r_S.items(), key=lambda a: a[1][1], reverse=True)
            # print("ans:" + str(ans))
            # if ans[0][0]:
            print(ans[0][0])
            answers.append(ans[0][0])
            hit_probs.append(ans[0][1][1])
            num_fp += 1
            # else:
            # answers.append(NULL)
            # hit_probs.append()
        # num_tp += 1

    # num_hits.append(num_tp)

    return hit_probs, answers, num_fp

def get_histo_v3(sa1_prob, keyword, hs, hn):

    tot_test = len(sa1_prob)

    incorrect_sa1_prob = []
    # correct_sa1_prob = []
    # final_TP = []
    # final_FN = []

    print("Starting the testing for the fixed keyword matching for: ")
    print("hspike: " + str(hs) + " hnode: " + str(hn))

    print("PLS keyword: " + str(len(keyword)))

    # num_tp = 0
    # num_fn = 0
    # num_fp = 0
    indices = []
    num_tp_fp = 0

    max_phones = []
    min_phones = []

    # print("Checking for True Positive and False Negative among 168 sentences where the keyword is present in each sentence:")

    # print(tot_test)
    # for i in range(tot_test):
    #     test_ex = sa1_prob[i][0].T
    #     p, tf, r_seq, in_seq  = testing_v3(test_ex, hs, keyword, hn)

    #     print("Test #" + str(i), end=" ")

    #     ans = sorted(r_seq.items(), key=lambda a: a[1][0], reverse=True)
    #     if bool(r_seq):
    #         num_tp += 1
    #         correct_sa1_prob.append(ans[0][1][0])
    #         print("Answer: " + str(ans[0][0]))
    #     else:
    #         num_fn += 1

    # print("#True Positive: " + str(num_tp) + " #False Negative: " + str(num_fn))

    print("Total Hits among 1344 sentences")
    # print("Checking for False Positive among 1344 sentences")
    for i in range(tot_test):
        # print("Testing #" + str(i))
        test_ex = sa1_prob[i].T
        test_ex = softmax(test_ex)
        # print(test_ex.shape)
        p, tf, r_seq, in_seq = testing_v3(test_ex, hs, keyword, hn)
        ans = sorted(r_seq.items(), key=lambda a: a[1][0], reverse=True)
        if bool(r_seq):
            print("Test #" + str(i), end=" ")
            num_tp_fp += 1
            # print(ans[0][1])
            # max_phones.append(ans[0][1][3])
            # min_phones.append(ans[0][1][4])
            incorrect_sa1_prob.append(ans[0][1][0])
            print("Answer: " + str(ans[0][0]))
            # indices.append(i)

    print("#False Positive + #True Positive: " + str(num_tp_fp))


    print("Complete....")

    return num_tp_fp, incorrect_sa1_prob #, max_phones, min_phones

    # return num_tp, num_fn, correct_sa1_prob
