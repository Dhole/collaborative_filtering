#! /bin/bash

echo -n "" > scale_res2.txt
# N coeff, 50000 nodes, 1% conn
./mega_graph.py 50000 0.01
for coeff in $(seq 10 10 100)
do
    cat _coeff_1000.txt | cut -d " " -f1-$coeff > coeff.txt
    ./cheby > tmp_out.txt
    echo 50000 0.01 $coeff >> scale_res2.txt
    cat tmp_out.txt | grep "Finished in" >> scale_res2.txt
    cat tmp_out.txt | grep "Final Runtime" >> scale_res2.txt
done

# N conn, 50000 nodes, 64 coeff
for conn in $(seq 0.005 0.005 0.05)
do
    ./mega_graph.py 50000 $conn
    cat _coeff_1000.txt | cut -d " " -f1-64 > coeff.txt
    ./cheby > tmp_out.txt
    echo 50000 $conn 64 >> scale_res2.txt
    cat tmp_out.txt | grep "Finished in" >> scale_res2.txt
    cat tmp_out.txt | grep "Final Runtime" >> scale_res2.txt
done


# 1% conn, N nodes, 64 coeff
for nodes in $(seq 5000 5000 50000)
do
    ./mega_graph.py $nodes 0.01
    cat _coeff_1000.txt | cut -d " " -f1-64 > coeff.txt
    ./cheby > tmp_out.txt
    echo $nodes 0.01 64 >> scale_res2.txt
    cat tmp_out.txt | grep "Finished in" >> scale_res2.txt
    cat tmp_out.txt | grep "Final Runtime" >> scale_res2.txt
done

