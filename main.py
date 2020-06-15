import json
import pickle
import pandas as pd 
import numpy as np
from PLS.process_lattice import *
from PLS.phone_lattice_scanning import *
from PLS.utils import *
from PLS.testing import *
# from rnn_kws_new.dl_model import *

if __name__ == "__main__":

	ids_fptr = open('PLS/probs/IDS Probabilities/GRU_5_384_79_probs.pkl', 'rb')
	ids_probs = pickle.load(ids_fptr)
	
	sa_fptr = open('PLS/probs/LSTM Probabilities/SA_res.pkl', 'rb')
	sa_prob = pickle.load(sa_fptr)

	# This should be the range of hspike and hnode
	# h_spike = np.arange(0.1, 0.5, 0.05)

	# h_node = [5e-8, 5e-7, 5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1]

	hspike = 0.25
	hnode = 5e-7

	# Following is the phone sequence of darksuit
    
	keyword_phones = ["pau", "d", "aa", "r", "pau", "k", "s", "uw", "dx"]

	keyword = search_phone(keyword_phones)

	timestamp_1 = time.time()

	results = []

	print("[" + str(time.time() - timestamp_1) + "]", end="")
	print("Start of the scanning for keyword: " + str(keyword))

	LSTM_output = sa_prob[0][0][0]

	answers = test_one_utterance(LSTM_output, ids_probs, [keyword], hspike, hnode, 1, 3, 20, True)
	results.append(answers)

	timestamp_2 = time.time()

	print("[" + str(timestamp_2 - timestamp_1) + "]", end="")
	print("End of the scanning...")

	print(results[0])
