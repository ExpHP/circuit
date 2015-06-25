set -e

clang++ -O2 --std=c++11 graph/cyclebasis/test_xorbasis.cpp -o graph/cyclebasis/test_xorbasis
graph/cyclebasis/test_xorbasis

./build-xorbasis.sh

python3 resistances.py -S bigholes -D remove -t 2 -s 6 -j 2 16June2015/hb100_100.gpickle -v
python3 resistances.py -S bigholes -D remove -t 2 -s 6 -j 2 mos2_65_65.gpickle -v --cyclebasis mos2_65_65.cyclebasis
python3 mos2.py -o derp.gpickle 50 50 -v -C derp.cyclebasis
echo 'done'
