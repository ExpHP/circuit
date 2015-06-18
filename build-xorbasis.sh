set -e
CFILEBASE=graph/cyclebasis/xorbasis
CYTHONBASE=graph/cyclebasis/cXorBasis
#CPPFLAGS="-Wall --pedantic -O2 --std=c++11 -fPIC -I/usr/include/python3.4"
CPPFLAGS="-Wall --pedantic -O2 --std=c++11 -fPIC -I/usr/include/python3.4 -DNDEBUG"
cython ${CYTHONBASE}.pyx --cplus
clang++ $CPPFLAGS -c ${CFILEBASE}.cpp -o ${CFILEBASE}.o
clang++ $CPPFLAGS -c ${CYTHONBASE}.cpp -o ${CYTHONBASE}.o
clang++ $CPPFLAGS --shared -o ${CYTHONBASE}.so ${CFILEBASE}.o ${CYTHONBASE}.o
echo 'done'
