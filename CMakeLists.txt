project(GraphLab)
# include(CheckCXXSourceCompiles)



# Build als
add_graphlab_executable(als als.cpp)
requires_eigen(als) # build and attach eigen

add_graphlab_executable(sparse_als sparse_als.cpp)
requires_eigen(sparse_als) # build and attach eigen

add_graphlab_executable(wals wals.cpp)
requires_eigen(wals) # build and attach eigen

add_graphlab_executable(sgd sgd.cpp)
requires_eigen(sgd) # build and attach eigen

add_graphlab_executable(biassgd biassgd.cpp)
requires_eigen(biassgd) # build and attach eigen

add_graphlab_executable(svdpp svdpp.cpp)
requires_eigen(svdpp) # build and attach eigen

add_graphlab_executable(svd svd.cpp)
requires_eigen(svd) # build and attach eigen


add_graphlab_executable(nmf nmf.cpp)
requires_eigen(nmf) # build and attach eigen

add_graphlab_executable(make_synthetic_als_data
  make_synthetic_als_data.cpp)
requires_eigen(make_synthetic_als_data) # build and attach eigen

add_graphlab_executable(knn knn.cpp)
requires_eigen(knn) # build and attach eigen

add_graphlab_executable(knn2 knn2.cpp)
requires_eigen(knn2) # build and attach eigen

add_graphlab_executable(knn3 knn3.cpp)
requires_eigen(knn3) # build and attach eigen

add_graphlab_executable(local_calc local_calc.cpp)
requires_eigen(local_calc) # build and attach eigen

add_graphlab_executable(local_calc_precomp local_calc_precomp.cpp)
requires_eigen(local_calc) # build and attach eigen

add_graphlab_executalbe(cheby cheby.cpp)
