# DONE

import sys
import numpy as np
import math
import collections
from collections import defaultdict, Counter
import json

NEG_INF = -float('inf')
INF = sys.maxsize
logarithm_mat = np.log


'''
Binary search function used in process lattice Level 2.
'''

def binary_search(arr, prob, x):
    l = 0
    r = arr.shape[0]-1
    while(l<r):
        mid = l + (r-l) // 2;

        if prob[arr[mid]] == x:
            return mid

        elif prob[arr[mid]] < x:
            l = mid + 1

        else:
            r = mid - 1
    return l

'''
Function to print lattice in a particular format to increase readability.
'''

def print_lattice(matrix):
    s = [[str(e) for e in row] for row in matrix]
    lens = [max(map(len, col)) for col in zip(*s)]
    fmt = '\t'.join('{{:{}}}'.format(x) for x in lens)
    table = [fmt.format(*row) for row in s]
    print('\n'.join(table))


def cmp(a, b):
    return (a==b)


def reverse_sublist(lst):
    N = len(lst)
    for n in range(N):
        lst[n] = lst[n][::-1]
    return lst


def make_new_beam():
    fn = lambda : (NEG_INF, NEG_INF)
    return collections.defaultdict(fn)


def logsumexp(*args):
    if all(a == NEG_INF for a in args[0]):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a-a_max) for a in args))
    return a_max + lsp

def logsumexp_v2(*args):
    sum_t = 0
    for i in range(len(args)):
        if args[i] != NEG_INF:
            sum_t += math.exp(args[i])

    if sum_t != 0:
        lsp = math.log(sum_t)
    else:
        lsp = NEG_INF

    return lsp

'''
This is the cost function. There are few options that were tried for dynamic PLS.
'''

def cost_func(prob, ed, arg):
    if arg == 0:
        return 40*ed*((-20)*prob + 0.005)
    elif arg == 1:
        return -prob*ed
    elif arg == 2:
        return logsumexp(prob,ed)
    elif arg == 3:
        return ed
    elif arg == 4:
        return 40*ed*((-20)*prob - 0.00005)
    elif arg == 5:
        return ed+prob  #(np.log(40) + ed + logsumexp_v2(np.log(20) + prob, np.log(0.00005)))
    elif arg == 6:
        return prob
    elif arg == 7:
        return 4*ed+prob
    else:
        return 400*ed

'''
This function adds log probabilities
'''

def prob_func(*args):

    if all(a == NEG_INF for a in args):
        return NEG_INF
    a_max = max(args)
    lsp = math.log(sum(math.exp(a-a_max) for a in args))
    return a_max + lsp


def make_seq_arr(N):
    # probability, edit distance wrt keyword, cost, DP
    fn = lambda : (NEG_INF, INF, INF, np.zeros(N+1))
    return collections.defaultdict(fn)

def make_record_seq():
    # prob, ed, cost, start time frame, end time frame
    fn = lambda : (NEG_INF, INF, INF, 0, 0)
    return collections.defaultdict(fn)


def make_seq_arr_fixed():
    # probability of sequence, time frame, pc
    fn = lambda : (NEG_INF, 0, 0)
    return collections.defaultdict(fn)

def make_seq_arr_fixed_key(N):
    #probability, time frame, pc for each keyword: -1 will indicate that the sequence is not valid for that keyword
    fn = lambda : (NEG_INF, 0, [0]*N)
    return collections.defaultdict(fn)

def make_seq_arr_fixed_key_v2(N):
    # prob, time frame, pc, max, min
    fn = lambda : (NEG_INF, 0, [0]*N, NEG_INF, 0)
    return collections.defaultdict(fn)

def compute_len_Range(str, ids_prob, top_k):
    N = len(str)
    start = N - top_k

    if start < 3:
        start = N

    end = N + top_k
    return start, end

def softmax(lstm_output):
    S, T = lstm_output.shape
    lstm_log_prob = []

    for t in range(T):
        temp = []
        deno = logsumexp_v2(lstm_output[:,t])

        for n in range(S):
            temp.append(lstm_output[n,t] - deno)
        lstm_log_prob.append(temp)

    lstm_log_prob = np.array(lstm_log_prob)

    return lstm_log_prob.T


def search_phone(phones):

    with open('PLS/phone_mapping.json') as f:
        phone_id = json.load(f)

    phones_ID_list = []

    for i in range(len(phones)):
        phones_ID_list.append(phone_id[phones[i]][0])

    return phones_ID_list