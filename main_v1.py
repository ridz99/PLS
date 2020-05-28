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

	ids_pickle = open("GRU_5_384_79_probs.pkl", "rb")
	ids_prob = pickle.load(ids_pickle)

	lstm_prob_pickle = open("SA_res.pkl", "rb")
	lstm_prob = pickle.load(lstm_prob_pickle)

	probs = lstm_prob[0]

	hspike = 0.1
	hnode = 0.0000005

	all_possible_answers = []

	# darksuit
	key_1 = [27, 7, 0, 28, 27, 19, 29, 34, 9]

	# choose the cost function argument referring the cost_func
	cost_arg = 4 # cost_func = edit_distance

	# trial values
	Smax = -3
	V = 25

	tot_examples = len(probs)

	timestamp_1 = time.time()

	print("[" + str(time.time() - timestamp_1) + "]", end="")
	print("Start of the scanning for keyword: " + str(key_1))

	'''
	uncomment the following block to run the tests over all sentences
	'''

	# for i in range(tot_examples):
	# 	example_prob = probs[i][0].T
	# 	print("[" + str(time.time() - timestamp_1) + "]", end="")
	# 	print("Test sentence #"+str(i))
	# 	final_lattice, time_frame, seq = testing_v2(example_prob, hspike, True, key_1, hnode, ids_prob, cost_arg)
	# 	ans = sorted(seq.items(), key=lambda a: a[1][2])
	# 	all_possible_answers.append((ans[0], i))

	#comment the following while uncommenting the above block
	#start
	example_prob = probs[0][0].T
	print(example_prob.shape)
	
	final_lattice, time_frame, seq = testing_v2(example_prob, hspike, True, key_1, hnode, ids_prob, cost_arg, Smax, V)
	ans = sorted(seq.items(), key=lambda a: a[1][2])
		
	all_possible_answers = ans
	for a in ans:
		print(a[0])

	#end	

	timestamp_2 = time.time()

	print("[" + str(timestamp_2 - timestamp_1) + "]", end="")
	print("End of the scanning...")

	print("Storing the results...")
	seq_test_sa_key_1 = open("seq_test_sa_key_1.pkl", "wb")
	pickle.dump(all_possible_answers, seq_test_sa_key_1)

	print("Following is the best possible answer:")
	all_possible_answers.sort(key=lambda a: a[1][2], reverse=True)
	print(all_possible_answers[0])    