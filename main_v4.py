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
import json

# This function is to find probabilities of the 80 keywords

if __name__ == "__main__":

	with open('probs/SA_res.pkl','rb') as f:
		probs = pickle.load(f)

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
	hn = h_node[1]

	count = 0
	# with open('keywords.json') as fl:
	# 	data = json.load(fl)
	#
	# fll = open("keywords.txt", "r")
	# key_words = fll.read()
	# x = key_words.split("\n")

	# darksuit, greasy, washwater, suit, oily, dark, wash, water
	keywords = [[27, 7, 0, 28, 27, 19, 29, 34, 27, 31],
				[27, 14, 28, 17, 29, 17, 36, 1, 30],
				[],
				[29, 34, 27, 31],
				[25, 20, 17],
				[27, 7, 0, 28, 27, 19],
				[36, 0, 30], 
				[36, 0, 27, 31, 11]]

	for key in keywords[:1]:

		print("[" + str(time.time() - timestamp_1) + "]", end="")
		# print(type(key))
		print("Start of the scanning for keyword: " + str(key))

		results_fixed_keyword.append([])
		indices.append([])

		prob_res = testing_all('results/pickle/lattices/final_lattice_25_7_SA.pkl', probs[0], key, hs, hn)
		results_fixed_keyword[count].append(prob_res)
		# print("Number of hits: " + str(fp_res))
		print("[" + str(time.time() - timestamp_1) + "]", end="")
		# print("Answer: " + str(prob_res))
		print("Completed this keyword")
		print("----------------------------------------------------------------")
		# if count%10 == 0:
		# 	print("Dumping for count: " + str(count))
		# 	fixed_res = open("results/pickle/all_probs_keywords_dmpls_darksuit.pkl", "wb")
		# 	pickle.dump(results_fixed_keyword, fixed_res)

		count += 1

	timestamp_2 = time.time()

	print("[" + str(timestamp_2 - timestamp_1) + "]", end="")
	print("End of the scanning...")

	# ind_res = open("Fixed_Results_Dark_Ind.pkl", "wb")
	# pickle.dump(indices, ind_res)

	# fixed_res = open("results/pickle/all_probs_keywords_dmpls_darksuit_TP.pkl", "wb")
	# pickle.dump(results_fixed_keyword, fixed_res)
