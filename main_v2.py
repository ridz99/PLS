import pickle
import numpy as np
import time
from itertools import groupby
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import animation
from phone_lattice_scanning import *
from utils import *
from process_lattice_L2_hnode import *
from edit_distance import *
from testing import *


if __name__ == "__main__":

	with open('probs/test_res.pkl','rb') as f:
		data = pickle.load(f)

	lstm_prob_pickle = open("probs/SA_res.pkl", "rb")
	lstm_prob = pickle.load(lstm_prob_pickle)

	sa1_log_prob = lstm_prob[0][::2] # to get the even indices
	sa2_log_prob = lstm_prob[0][1::2] # to get the odd ones

	# darksuit

	# The sequence is in the order of most frequent to least frequent
	# key_1_freq = [63, 46, 21, 16]
	# darksuit
	key_darksuit = [[27, 7, 0, 28, 27, 19, 29, 34],
			 		[27, 7, 0, 28, 27, 19, 29, 34, 27, 31],
			 		[27, 7, 0, 28, 27, 19, 29, 34, 9],
			 		[27, 7, 0, 28, 27, 29, 34]]

	#greasy
	key_greasy = [[27, 14, 28, 17, 29, 17, 36, 1, 30],
			 	  [27, 14, 28, 17, 38, 33, 36, 1, 30]]

	#water
	key_water = [[]]
	#year
	key_year =[[]]
	#dark
	key_dark = [[27, 7, 0, 28, 27, 19]]
	#wash
	key_wash = [[36, 0, 30]]
	#suit
	key_suit = [[29, 34, 9], [29, 34, 27, 31]]
	#oily
	key_oily = [[25, 20, 17]]

	key_lst_1 = ['water', 'year', 'dark', 'wash', 'suit', 'oily', 'greasy', 'darksuit']

	# 0.9 is the last sort because the search is almost zero or 1 among 168 sentences
	h_spike = np.arange(0.1, 0.5, 0.05)

	# h_node = np.linspace(0.00000005, 0.5, 6, True)

	h_node = [5e-8, 5e-7, 5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1]

	m = len(h_spike[:6])
	n = h_node[1:2]

	# results_fixed_keyword = [[(0, tuple())]*n]*m
	results_fixed_keyword = []
	# indices = []

	timestamp_1 = time.time()

	print("[" + str(time.time() - timestamp_1) + "]", end="")
	print("Start of the scanning for keyword: suit")

	for i in range(m):
		hs = h_spike[i]
		results_fixed_keyword.append([])
		# indices.append([])
		for hn in n:
			print("[" + str(time.time() - timestamp_1) + "]", end="")
			# hn = h_node[j]
			# tp_res, fn_res, prob_res = get_histo_v3(sa2_log_prob, key_oily, hs, hn)
			fp_res, prob_res = get_histo_v3(data[0], key_suit, hs, hn)
			results_fixed_keyword[i].append((fp_res, prob_res))
			# indices[i].append(ind)

	# tp_res, fn_res, prob_res = get_histo_v3(sa1_log_prob, key_1, 0.4, 5e-4)

	# print(tp_res)

	timestamp_2 = time.time()

	print("[" + str(timestamp_2 - timestamp_1) + "]", end="")
	print("End of the scanning...")

	# ind_res = open("Fixed_Results_Dark_Ind.pkl", "wb")
	# pickle.dump(indices, ind_res)

	fixed_res = open("results/pickle/Fixed_Results_Suit_FP_7.pkl", "wb")
	pickle.dump(results_fixed_keyword, fixed_res)
