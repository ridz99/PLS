import numpy as np
from utils import binary_search
import pandas as pd

def generate_lattice(outputs, variable_nodes, threshold, blank_token_id=40, top_n=5):
    time_steps = outputs.shape[1]

    temp3 = outputs

    fl = np.argsort(temp3, axis=0)

    # print("generate_lattice:")
    # print("time_steps: " + str(time_steps))

    # print(fl)

    # variable_nodes is a flag that is used to whether consider dynamic number of nodes or go for fixed number of nodes at each time instance
    if variable_nodes == False:
        # print("Fixed nodes in the lattice")
        final_lattice = fl[-top_n:, :]
        final_lattice = final_lattice[::-1]
    else:
        # print("Variable nodes in the lattice")
        # the main idea is to keep all the phones which are above a certain threshold for the searching purpose
        threshold = np.log(threshold)

        ind = binary_search(fl[:,0], outputs[:,0], threshold)
        # print(ind)
        data = fl[ind:, 0][::-1].tolist()

        temp_lattice = pd.DataFrame(data, columns=[0])

        # print("time steps:" + str(time_steps))

        for t in np.arange(1, time_steps):
            ind = binary_search(fl[:,t], outputs[:,t], threshold)
            data = fl[ind:, t][::-1].tolist()
            df = pd.DataFrame(data, columns=[t])
            temp_lattice = pd.concat([temp_lattice, df], axis=1, join='outer')
        
        final_lattice = temp_lattice

    return final_lattice
