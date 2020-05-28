import numpy as np
from utils import *

'''
Edit distance required to convert string_1 into string_2
'''
def edit_distance(ids_prob, string_1, string_2):

    m, n = len(string_1), len(string_2)

    insert_prob, delete_prob, substitute_prob = ids_prob[0], ids_prob[1], ids_prob[2]

    insert_prob = np.array(logarithm_mat(insert_prob))
    # print(insert_prob.shape)
    delete_prob = np.array(logarithm_mat(delete_prob))
    # print(delete_prob.shape)
    substitute_prob = np.array(logarithm_mat(substitute_prob))

    dp = np.zeros((m + 1, n + 1))

    for i in range(m + 1):
        for j in range(n + 1):
            if i == 0 and j == 0:
                dp[i][j] = 0
            elif i == 0:
                test_list = [int(k) for k in string_2[:j]]
                dp[i][j] = np.sum(insert_prob[test_list])
            elif j == 0:
                test_list = [int(k) for k in string_1[:i]]
                dp[i][j] = np.sum(delete_prob[test_list])
            elif string_1[i - 1] == string_2[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
            else:

                delete, insert, substitute = delete_prob[int(string_1[i - 1])], insert_prob[int(string_2[j - 1])], substitute_prob[int(string_1[i - 1])][int(string_2[j - 1])]
                dp[i][j] = max(dp[i - 1][j] + delete,
                                dp[i][j - 1] + insert,
                                dp[i - 1][j - 1] + substitute)

    return dp[m][n]


'''
Following edit distance is the optimized one which uses O(n) time complexity using
the locality rule but this modification uses extra of O(n) space complexity in the 
PLS algorithm, where n is the length of the keyword.
'''


def edit_distance_v2(new_added_Phone, target_Seq, last_column_ED, IDS_Prob):

    '''
    if it is the first time the edit distance is calculated then the last_column_ED
    will be an array of zeroes which is passed at the calling point
    '''
    insert_prob, delete_prob, substitute_prob = IDS_Prob[0], IDS_Prob[1], IDS_Prob[2]

    insert_prob = np.array(logarithm_mat(insert_prob))
    # print(insert_prob.shape)
    delete_prob = np.array(logarithm_mat(delete_prob))
    # print(delete_prob.shape)
    substitute_prob = np.array(logarithm_mat(substitute_prob))


    n = len(target_Seq)

    dp = np.zeros(n+1)

    for i in range(n+1):
        if i == 0:
            # print("ED v2:" + str(delete_prob[new_added_Phone]))
            # dp[i] = prob_func(last_column_ED[i], insert_prob[new_added_Phone])
            dp[i] = last_column_ED[i] + insert_prob[new_added_Phone]

        elif new_added_Phone == target_Seq[i-1]:
            dp[i] = last_column_ED[i-1]

        else:

            delete, insert, substitute = delete_prob[new_added_Phone], insert_prob[int(target_Seq[i-1])], substitute_prob[new_added_Phone][int(target_Seq[i-1])]

            dp[i] = max(last_column_ED[i] + delete,
                        dp[i-1] + insert,
                        last_column_ED[i-1] + substitute)

            # dp[i] = max(prob_func(last_column_ED[i], delete),
            #             prob_func(dp[i-1], insert),
            #             prob_func(last_column_ED[i-1], substitute))

    return dp
