import numpy as np
from utils import logsumexp_v2

def softmax(lstm_output):
    S, T = lstm_output.shape
    # print("softmax:", end=" ")
    lstm_log_prob = []

    for t in range(T):
        temp = []
        deno = logsumexp_v2(lstm_output[:,t])

        for n in range(S):
            temp.append(lstm_output[n,t] - deno)
        lstm_log_prob.append(temp)

    lstm_log_prob = np.array(lstm_log_prob)

    # print(lstm_log_prob.T.shape)

    return lstm_log_prob.T