import json

def read_json_80():
    with open('keywords.json') as f:
        data = json.load(f)

    f = open("keywords.txt", "r")
    key_words = f.read()
    x = key_words.split("\n")

    return x, data
