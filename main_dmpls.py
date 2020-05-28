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

# this main script is used to test the DMPLS method

if __name__ == "__main__":

    # darksuit, greasy, washwater, suit, oily, dark, wash, water
    keywords = [[27, 7, 0, 28, 27, 19, 29, 34, 27, 31],
                [27, 14, 28, 17, 29, 17, 36, 1, 30],
                [],
                [29, 34, 27, 31],
                [25, 20, 17],
                [27, 7, 0, 28, 27, 19],
                [36, 0, 30],
                []]
