collaborative_filtering
=======================

Graphlab Collaborative Filtering

Added K-Nearest Neighbours filtering (Works in 3 steps: knn, knn2, knn3)

local_cacl is able to gather information of the full subgraph surrounding one vertex, 
    with the local subgraph it will try to predict the rating value of the vertex using
    the method proposed in ICASSP2013 [1]

fold_cross_validation.py is a tool to separate a database by users for a fold cross validation test

[1] SIGNALPROCESSING TECHNIQUES FOR INTERPOLATION IN GRAPHDATA by Sunil K Narang, Akshay Gadde and Antonio Ortega
