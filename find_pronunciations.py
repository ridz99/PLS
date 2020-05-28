'''
This script is used to find the specific phone sequence of a word as noted in the TIMIT
dataset.
'''


import os
import json
import pickle

def search(name, list):
    for i in range(len(list)):
        for j in range(len(list[i])):
            if name == list[i][j]:
                return i
    return -1

def search_phone(name, list):
    for i in range(len(list)):
        if name == list[i]:
            return i


def find(name, path):
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)

def find_all(name, path):
    result = []
    for root, dirs, files in os.walk(path):
        if name in files:
            result.append(os.path.join(root, name))
    return result

final_list_files_WRD = []

with open('keywords.json') as f:
    data = json.load(f)

f = open("keywords.txt", "r")
key_words = f.read()
x = key_words.split("\n")


with open('folding_2.json') as f:
    folding = json.load(f)

fold_key = []
fold_values = []

for k, v in folding.items():
    fold_key.append(k)
    fold_values.append(v)


with open('phone_mapping.json') as f:
    phones = json.load(f)

phone_keys = []

for k in phones.keys():
    phone_keys.append(k)

pronunciation_list = []

for i in range(len(data)):
    final_list_files_WRD.append([])
    pronunciation_list.append([])

    for j in range(len(data[x[i]][0])):
        if data[x[i]][1][j] != 0:
            y = data[x[i]][0][j] + '.WRD'

            # if data[x[i]][1][j] == 1:
            tmp_all = find_all(y, '../TIMIT/TEST')
            for tmp in tmp_all:
                f_tmp = open(tmp, 'r')
                lines = f_tmp.read()
                lines = lines.split("\n")
                for k in range(len(lines)):
                    temp = lines[k].split(" ")
                    if len(temp) > 2:
                        if temp[2] == x[i]:
                            print(temp[2], x[i])
                            start = int(temp[0])
                            end = int(temp[1])
                            break

                tmp = tmp[:-3] + 'PHN'
                f_tmp = open(tmp, 'r')
                lines = f_tmp.read()
                lines = lines.split("\n")
                pron = []
                pron_num = []
                for k in range(len(lines)):
                    temp = lines[k].split(" ")
                    if len(temp) > 2:
                        if int(temp[0]) >= start:
                            pos = search(temp[2], fold_values)
                            if pos != -1:
                                pron.append(fold_key[pos])
                                pron_num.append(search_phone(fold_key[pos], phone_keys))
                            else:
                                pron.append(temp[2])
                                pron_num.append(search_phone(temp[2], phone_keys))

                        if int(temp[1]) == end:
                            # pron.append(temp[2])
                            break

                pronunciation_list[i].append(pron_num)

    print("Number of pronunciations for keyword " + x[i] + " " + str(len(pronunciation_list[i])))

f = open("pronunciations.pkl", "wb")
pickle.dump(pronunciation_list, f)
