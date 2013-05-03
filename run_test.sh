#! /bin/bash

CROSS_PATH="/home/esanou/tmp/ml-100k/cross_validation"
MV_PATH="movielens"

rm $MV_PATH
ln -s "movielens_cross" $MV_PATH

for i in {0..4}
do
    echo "Test #$i"
    rm $MV_PATH/*
    ln -s $CROSS_PATH/u$i.train $MV_PATH/u$i.train
    ln -s $CROSS_PATH/u$i.test $MV_PATH/u$i.validate
    ./knn
    ./knn2
    ./local_calc
    cat out_res_* > out_res.$i
done
