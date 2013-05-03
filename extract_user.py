#! /usr/local/bin/python2.7

import sys

user_id = sys.argv[1]

f1 = "out_test_rat_1_of_4"
f2 = "out_test_rat_2_of_4"
f3 = "out_test_rat_3_of_4"
f4 = "out_test_rat_4_of_4"
ff = [f1, f2, f3, f4]



for file in ff:
    with open (file, 'r') as f:
        for line in f:
            words = line.split(" ")
            for n in range(0, len(words)):
                if words[n] == user_id:
                    print words[0], words[n], words[n+1]
