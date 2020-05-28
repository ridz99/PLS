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

# this file dumps the final lattice for the 1344 sentences at specified hspike and hnode value

if __name__ == "__main__":

    with open('probs/SA_res.pkl','rb') as f:
	       probs = pickle.load(f)

	# 0.9 is the last sort because the search is almost zero or 1 among 168 sentences
    h_spike = np.arange(0.1, 0.5, 0.05)

	# h_node = np.linspace(0.00000005, 0.5, 6, True)
    h_node = [5e-8, 5e-7, 5e-6, 5e-5, 5e-4, 5e-3, 5e-2, 5e-1]

    m = len(h_spike[:6])
    n = h_node[1:2]

    hs = h_spike[3]
    hn = h_node[1]

    final_lattices = []

    N = len(probs[0])

    for i in range(N):
        print("#"+str(i))
        test_ex = probs[0][i][0].T
        # test_ex = softmax(test_ex)
        key = []
        fl, tf = testing_v3(test_ex, hs, key, hn)
        final_lattices.append((fl, tf))

        if i%10 == 0:
            print("Dumping at count: " + str(i))
            f = open("results/pickle/final_lattice_25_7_SA.pkl", "wb")
            pickle.dump(final_lattices, f)

    f = open("results/pickle/final_lattice_25_7_SA.pkl", "wb")
    pickle.dump(final_lattices, f)
