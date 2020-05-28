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
from process_lattice_L1_hspike import *
from generate_lattice import *
from softmax import *
import json


def test_one_utterance(LSTM_prob, phone_lattice, time_frames, keyword, ids_prob, PLS_Arg, cost_Arg=5, V=20):

	phone_lattice = phone_lattice.values

	num_tp = 0

	if PLS_Arg == 1:
		recorded_seq = Fixed_PLS_single_pronunciation(phone_lattice, time_frames, LSTM_prob, keyword)
	elif PLS_Arg == 2:
		recorded_seq = Fixed_PLS_multi_pronunciation(phone_lattice, time_frames, LSTM_prob, keyword)
	else:
		recorded_seq = Modified_Dynamic_PLS(LSTM_prob, phone_lattice, keyword, time_frames, ids_prob, cost_Arg, V)

	if bool(recorded_seq):
		num_tp += 1
		ans = sorted(recorded_seq.items(), key = lambda a: a[1][1], reverse=True)
		print("Match: " + ans[0][0], end=" ")
		print("Probabilty: " + ans[0][1][1])

	return num_tp


def test_all_utterances(LSTM_prob, filename, keyword, ids_prob, PLS_Arg, cost_Arg=5, V=20):

	final_lattices, tf = read_pickle_lattice_file(filename)

	n_test = len(final_lattices)
	# print(tf)
	# print(len(LSTM_prob))

	num_tp = 0
	# num_fp = 0

	answers = []
	hits_probs = []

	for i in range(n_test):

		log_prob = LSTM_prob[i][0].T
		log_prob = np.log(log_prob)

		phone_lattice = final_lattices[i]
		time_frames = tf[i]


		# if the phone_lattice is read from the pkl file then keep this line otherwise comment it
		phone_lattice = phone_lattice.values
		# print(phone_lattice.shape)

		if PLS_Arg == 1:
			recorded_seq = Fixed_PLS_single_pronunciation(phone_lattice, time_frames, log_prob, keyword[0])
		elif PLS_Arg == 2:
			recorded_seq = Fixed_PLS_multi_pronunciation(phone_lattice, time_frames, log_prob, keyword)
		else:
			recorded_seq = Modified_Dynamic_PLS(log_prob, phone_lattice, keyword[0], time_frames, ids_prob, cost_Arg, V)


		if bool(recorded_seq):
			num_tp += 1
			ans = sorted(recorded_seq.items(), key = lambda a: a[1][1], reverse=True)
			
			answers.append(ans[0][0])
			hits_probs.append(ans[0][1][1])

			# print("Match: " + ans[0][0], end=" ")
			# print("Probabilty: " + ans[0][1][1])

	return answers, hits_probs, num_tp


def form_lattices(LSTM_output, hspike, hnode, variable = True):

	log_prob = np.log(LSTM_output)

	phone_lattice_raw, time_frame = process_lattice_hspike(log_prob, hspike)

	final_lattice = generate_lattice(phone_lattice_raw, variable, hnode)

	if variable:
		final_lattice, time_frame = process_lattice_v3(final_lattice, time_frame, log_prob)
	else:
		final_lattice, time_frame = process_lattice(final_lattice, time_frame)

	return final_lattice, time_frame


def form_lattices_for_all_utterances(LSTM_prob, hspike, hnode):

	N = len(LSTM_prob[0])
	final_lattices = []

	for i in range(N):

		text_Ex = LSTM_prob[0][i][0].T
		fl, tf = form_lattices(LSTM_prob, hspike, hnode)
		final_lattices.append((fl, tf))

		if i%10:
			f = open("results/probs/lattices/final_lattice_25_7_SA.pkl", "wb")
			pickle.dump(final_lattices, f)

	f = open("results/probs/lattices/final_lattice_25_7_SA.pkl", "wb")
	pickle.dump(final_lattices, f)

##########################################################################################

if __name__ == "__main__":

	with open('probs/SA_res.pkl','rb') as f:
		probs = pickle.load(f)

	ids_fptr = open('probs/GRU_5_384_79_probs.pkl', 'rb')
	ids_probs = pickle.load(ids_fptr)

	hs, hn = 0.25, 5e-8

	lattice_filename = "results/pickle/lattices/final_lattice_25_8_SA.pkl"

	h_spike = np.arange(0.1, 0.5, 0.05)

	h_node = [5e-8, 5e-7, 5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1]

	hs = h_spike[3]
	hn = h_node[0]

	# if the lattices are not formed then uncomment the following line
	# form_lattices_for_all_utterances(probs, hs, hn)

	# if you want to use the 80 keywords testing use uncomment the following code

	# with open('keywords.json') as fl:
	# 	keywords = json.load(fl)
	
	# fll = open("keywords.txt", "r")
	# key_words = fll.read()
	# x = key_words.split("\n")

	# pass key as keywords[x[i]][3]	
	# N = len(keywords)

	#end

	# this is one pronunciation per keyword
	# darksuit, greasy, suit, oily, dark, wash, water
	keywords = [[27, 7, 0, 28, 27, 19, 29, 34, 27, 31],
				[27, 14, 28, 17, 29, 17, 36, 1, 30],
				[29, 34, 27, 31],
				[25, 20, 17],
				[27, 7, 0, 28, 27, 19],
				[36, 0, 30], 
				[36, 0, 27, 31, 11]]

	N = len(keywords)


	timestamp_1 = time.time()

	results = []

	count = 0

	for key in range(N):
		print("[" + str(time.time() - timestamp_1) + "]", end="")
		print("Start of the scanning for keyword: " + str(keywords[key]))

		results.append([])

		answers, ans_probs, num_tp = test_all_utterances(probs[0], lattice_filename, [keywords[key]], ids_probs, 1)

		results[count].append((answers, ans_probs, num_tp))

		print("[" + str(time.time() - timestamp_1) + "]", end="")
		print("Completed this keyword")

		count += 1

	timestamp_2 = time.time()

	print("[" + str(timestamp_2 - timestamp_1) + "]", end="")
	print("End of the scanning...")

	# fixed_res = open("results/pickle/filename.pkl", "wb")
	# pickle.dump(results_fixed_keyword, fixed_res)