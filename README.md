collaborative_filtering
=======================

Graphlab Collaborative Filtering

Added K-Nearest Neighbours filtering (Works in 3 steps: knn, knn2, knn3)

local_calc is able to gather information of the full subgraph surrounding one vertex, 
    with the local subgraph it will try to predict the rating value of the vertex using
    the method proposed in ICASSP2013 [1]

fold_cross_validation.py is a tool to separate a database by users for a fold cross validation test

precompute_local will find the laplacian and SVD decomposition of each graph made with all the movies
    rated by the same user. It will save the results in a file called "out_eigen_".
    The file structure is as follows:
    userID n_rated_movies n_eiven_values movieID1 sig_min1 movieID2 sig_min2 movieID3 sig_min3 ...
    eigen_values(1) eigen_values(2) eigen_values(3) ...
    eigen_vectors(0,0) eigen_vectors(0,1) eigen_values(0,2) ... eigen_vales(1,0) eigen_values(1,1) eigen_values(1,2) ...

local_calc_precomp is symmilar to local_calc, but instead of computing the Laplacian matrix and it's SVD
    decompositio for each user rating, it will preload the SVD decomposition from the file "out_eigen_".
    This should speed the process.


TODO:
    - Save the eigen vectors and eigen values in binary format instead of text format so that we get smaller files
    - Use functions in the code to do the rating prediction with matrix as inputs (so that the algebra computation
        can be easily identified)
    - Enable multi-threading in precompute_local (using a thread pool???)


[1] SIGNALPROCESSING TECHNIQUES FOR INTERPOLATION IN GRAPHDATA by Sunil K Narang, Akshay Gadde and Antonio Ortega
