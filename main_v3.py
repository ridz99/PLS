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

# This function is to find the max and min probabilities in the phone sequence for short keywords

if __name__ == "__main__":

	with open('probs/test_res.pkl','rb') as f:
		data = pickle.load(f)

	lstm_prob_pickle = open("probs/SA_res.pkl", "rb")
	lstm_prob = pickle.load(lstm_prob_pickle)

	sa1_log_prob = lstm_prob[0][::2] # to get the even indices
	sa2_log_prob = lstm_prob[0][1::2] # to get the odd ones

	key_list = {}

	#water
	key_list['key_water'] = [[36, 0, 27, 31, 11]]
	#year
	key_list['key_year'] =[[37, 16, 11]]
	#dark
	key_list['key_dark'] = [[27, 7, 0, 28, 27, 19]]
	#wash
	key_list['key_wash'] = [[36, 0, 30]]
	#suit
	key_list['key_suit'] = [[29, 34, 9], [29, 34, 27, 31]]
	#oily
	key_list['key_oily'] = [[25, 20, 17]]

	key_lst_name = ['water', 'year', 'dark', 'wash', 'suit', 'oily', 'greasy', 'darksuit']

	# 0.9 is the last sort because the search is almost zero or 1 among 168 sentences		 
	h_spike = np.arange(0.1, 0.5, 0.05)

	# h_node = np.linspace(0.00000005, 0.5, 6, True) 

	h_node = [5e-8, 5e-7, 5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1]

	m = len(h_spike[:6])
	n = h_node[1:2]

	results_fixed_keyword = []
	indices = []

	timestamp_1 = time.time()
	
	hs = h_spike[3]
	hn = h_node[0]

	count = 0

	for key in key_lst_name[:6]:

		print("[" + str(time.time() - timestamp_1) + "]", end="")
		# print(type(key))
		print("Start of the scanning for keyword: " + 'key_' + key)

		results_fixed_keyword.append([])
		indices.append([])

		tp_fp_res, prob_res, max_p_prob, min_p_prob = get_histo_v3(data[0], key_list['key_' + key], hs, hn)
		results_fixed_keyword[count].append((tp_fp_res, prob_res, max_p_prob, min_p_prob))
		print("[" + str(time.time() - timestamp_1) + "]", end="")
		count += 1

	timestamp_2 = time.time()

	print("[" + str(timestamp_2 - timestamp_1) + "]", end="")
	print("End of the scanning...")

	# ind_res = open("Fixed_Results_Dark_Ind.pkl", "wb")
	# pickle.dump(indices, ind_res)

	fixed_res = open("results/pickle/max_min_probs_6_keywords.pkl", "wb")
	pickle.dump(results_fixed_keyword, fixed_res)