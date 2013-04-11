#! /usr/bin/python3

import sys, os, random

filename = "u.data"
num_div = 5

filename = sys.argv[1]
num_div = int(sys.argv[2])

data = {}
with open(filename, 'r') as f:
    for line in f:
        val = line.split('\t')
        user_id = int(val[0])
        item_id = int(val[1])
        rating = int(val[2])
        
        if user_id not in data:
            data[user_id] = []
        data[user_id].append((item_id, rating))

num_usr = len(data)

os.mkdir("cross_validation")
os.chdir("cross_validation")

test  = {}
test[0] = ""
train = {}

keys = list(data.keys())
random.shuffle(keys)

ind = 0
n_usr_done = 0
for key in keys:
    for p in data[key]:
        test[ind] += str(key) + "\t" + str(p[0]) + "\t" + str(p[1]) + "\n"
    n_usr_done += 1
    if n_usr_done > (num_usr / num_div):
        n_usr_done = 0
        ind += 1
        test[ind] = ""

for i in range(ind + 1):
    train[i] = ""
    for j in range(ind + 1):
        if i != j:
            train[i] += test[j]
            
for i in range(ind + 1):
    with open("u" + str(i) + ".test", 'w') as f:
        f.write(test[i])
    with open("u" + str(i) + ".train", 'w') as f:
        f.write(train[i])
        
