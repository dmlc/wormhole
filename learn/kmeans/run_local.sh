nbr_node=$1
datan=rcv1_test.binary
#datan=agaricus.txt.train
datap=../data/$datan
nomp=$2
make
if [ ! -f $datap ];
then
  cd ../data
  wget http://www.csie.ntu.edu.tw/~cjlin/libsvmtools/datasets/binary/$datan.bz2
  bzip2 -d $datan.bz2
  cd ../kmeans
fi
../../dmlc-core/tracker/dmlc_local.py -n $nbr_node kmeans.dmlc $datap 40 5 $datap.out $nomp
