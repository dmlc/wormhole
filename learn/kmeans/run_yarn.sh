nbr_node=$1
datan=rcv1_test.binary
#datan=a3.txt
#datan=agaricus.txt.train
export fsprefix=hdfs://localhost:9000
datap=$fsprefix/user/wttian/libsvmdata/$datan
datapout=$fsprefix/user/wttian/libsvmdata/$datan.out
nomp=$2
make
if [ "$nbr_node" == "0" ]
then
  ./kmeans.dmlc $datap 40 5 $datapout $nomp
else
  ../../dmlc-core/tracker/dmlc_yarn.py -mem 4096 --log-level DEBUG --log-file log -n $nbr_node --vcores $nomp kmeans.dmlc $datap 40 10 $datapout $nomp
fi
