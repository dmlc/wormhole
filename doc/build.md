build problem:

1.How to make the project when the local gcc version does not support c++11 ?

First install the gcc version which support c++11.Then change the gcc version in the makefile. 
After that ,you can make the makefile.
when run the run-mushroom.sh,you should add --ship-libcxx parameter to ship the gcclib so 






