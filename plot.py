import matplotlib
matplotlib.use('Agg')
import pickle
import numpy as np
import math
import pandas as pd
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from matplotlib import animation
import sys
from matplotlib.ticker import (AutoMinorLocator, MultipleLocator)
import sys

# This is the correct one

if __name__ == "__main__":

	file_TP = open("results/pickle/Fixed_Results_Oily_TP.pkl", "rb")
	prob_TP = pickle.load(file_TP)

	file_FP = open("results/pickle/Fixed_Results_Oily_FP.pkl", "rb")
	prob_FP = pickle.load(file_FP)

	# hspike = 0.2 and hnode = 5e-8
	list_TP = prob_TP[2][0][1]
	list_FP = prob_FP[2][0][1]

	print("Keyword: Oily")
	print("#TP: " + str(len(list_TP)))
	print("#FP: " + str(len(list_FP)))

	# dist = pd.DataFrame(list(zip(list_TP, list_FP)), columns =['True Positive', 'False Positive'])
	# print(dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2))

	n_bins = 40
	fig, ax = plt.subplots()
	# ax.kde(legend=False, title='TP vs. FP: hspike=0.2, hnode=5e-8')
	ax.hist(list_TP, n_bins, histtype='stepfilled', alpha=0.5, color="green")
	ax.hist(list_FP, n_bins, histtype='stepfilled', alpha=0.5, color="red")
	ax.set_title("TP vs. FP, hspike=0.2, hnode=5e-8: Oily")
	# ax.set_ylim(0, 30)
	# ax.yaxis.set_major_locator(MultipleLocator(10))
	# ax.yaxis.set_minor_locator(AutoMinorLocator(5))
	ax.grid(which='major', color='#CCCCCC', linestyle='--')
	ax.set_xlabel('Probability')
	ax.grid(axis='y')
	ax.set_facecolor('#d8dcd6')
	fig.savefig('results/images/TP_vs_FP_oily_2_8.png')

# This is correct one but change the density with other parameter (frequency rather)

# if __name__ == "__main__":
#
# 	# argList = sys.argv
#
# 	file_TP = open("results/pickle/Fixed_Results_Dark_TP.pkl", "rb")
# 	prob_TP = pickle.load(file_TP)
#
# 	file_FP = open("results/pickle/Fixed_Results_Dark_FP.pkl", "rb")
# 	prob_FP = pickle.load(file_FP)
#
# 	# hspike = 0.2 and hnode = 5e-8
# 	list_TP = prob_TP[2][0][1]
# 	list_FP = prob_FP[2][0][1]
#
# 	print("Keyword: Suit")
# 	print("#TP: " + str(len(list_TP)))
# 	print("#FP: " + str(len(list_FP)))
#
# 	dist = pd.DataFrame(list(zip(list_TP, list_FP)), columns =['True Positive', 'False Positive'])
#
# 	print(dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2))
#
# 	fig, ax = plt.subplots()
# 	dist.plot.kde(ax=ax, legend=False, title='TP vs. FP: hspike=0.2, hnode=5e-8')
# 	dist.plot.hist(histtype='stepfilled', alpha=0.5, ax=ax, density=True)
#
# 	# ax.set_ylim(0, 30)
# 	# ax.yaxis.set_major_locator(MultipleLocator(10))
# 	# ax.yaxis.set_minor_locator(AutoMinorLocator(5))
#
# 	ax.grid(which='major', color='#CCCCCC', linestyle='--')
#
# 	ax.set_xlabel('Probability')
# 	ax.grid(axis='y')
# 	ax.set_facecolor('#d8dcd6')
# 	fig.savefig('results/images/TP_vs_FP_dark_2_8.png')


#This is another one for plotting the max and min of the probabilities of the detected sequence

# if __name__ == "__main__":
#
# 	file_prob = open("results/pickle/max_min_probs_6_keywords.pkl", "rb")
# 	data = pickle.load(file_prob)
#
# 	key_lst_name = ['water', 'year', 'dark', 'wash', 'suit', 'oily', 'greasy', 'darksuit']
#
# 	n = len(data)
#
# 	for i in range(n):
#
# 		print("Keyword: " + key_lst_name[i])
#
# 		list_max = data[i][0][2]
# 		list_min = data[i][0][3]
#
# 		dist = pd.DataFrame(list(zip(list_max, list_min)), columns = ['Max Prob', 'Min Prob'])
#
# 		print(dist.agg(['min', 'max', 'mean', 'std']).round(decimals=2))
#
# 		fig, ax = plt.subplots()
# 		dist.plot.kde(ax=ax, legend=False, title='max vs min: hspike=0.2, hnode=5e-8')
# 		dist.plot.hist(histtype='stepfilled', alpha=0.5, ax=ax)
#
# 		ax.set_ylim(0, 60)
# 		ax.yaxis.set_major_locator(MultipleLocator(10))
# 		ax.yaxis.set_minor_locator(AutoMinorLocator(4))
#
# 		ax.set_xlabel('Probability')
# 		ax.grid(axis='y')
# 		ax.set_facecolor('#d8dcd6')
# 		fig.savefig('results/images/max_vs_min_' + key_lst_name[i] + '.png')
