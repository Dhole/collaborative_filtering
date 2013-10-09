#! /usr/bin/env python

import sys, random

size = int(sys.argv[1])
conn = float(sys.argv[2])

print 'Generating signal...'
# write signal file
with open("graph_signal.txt", 'w') as f:
    for i in range(size):
        f.write(str(i + 1) + ' ' + str(random.uniform(0, 10)) + '\n')

# write topology file
#with open("graph_topology.txt", 'w') as f:
#    for i in range(size):
#        for j in range(i, size - 1):
#            node_a = i + 1
#            node_b = j + 2
#            #print node_a, node_b
#            if random.random() < conn:
#                wei = random.random()
#                f.write(str(node_a) + ' ' + str(node_b) + ' ' + '{0:.2f}'.format(wei) + '\n')
#                #f.write(str(node_b) + ' ' + str(node_a) + ' ' + '{0:.2f}'.format(wei) + '\n')

print 'Generating topology...'
links = set()
n_links = (conn * (size**2))
with open("graph_topology.txt", 'w') as f:
    while len(links) < n_links:
        a = random.randint(1,size)
        b = random.randint(1,size)
        #a = round(random.random()*(size-1))+1
        #b = round(random.random()*(size-1))+1
        if a != b:
            links.add((a,b))

    for link in links:
        wei = random.random()
        f.write(str(link[0]) + ' ' + str(link[1]) + ' ' + '{0:.2f}'.format(wei) + '\n')
