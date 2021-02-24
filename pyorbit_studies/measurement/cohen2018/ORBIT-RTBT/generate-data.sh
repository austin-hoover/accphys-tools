#!/bin/bash

# 
i=1
max=10
s=$(seq -f "%04g" 1 $max)
while [ $i -le $max ]
do
	# itNum=$i
	# stepSize="0.001"
	echo "rep number $i"
	suf=`printf "%04d" $i`
	dirname="y_optics_$suf"
	#print the lat file name to tmp
	echo "/Users/4tc/Dropbox/suli 2018 project/$dirname/latticelist.mu.lat" > "/tmp/whereIsLattice.txt"
	pyORBIT "/Users/4tc/Dropbox/suli 2018 project/ORBIT-RTBT/2018-06-26-madx-RTBT-pyorbit_r0.py" 
	# move results to folder
	mv ws*.dat "/Users/4tc/Dropbox/suli 2018 project/$dirname/"
	mv "/Users/4tc/Dropbox/suli 2018 project/ORBIT-RTBT/initial.mu.txt" "/Users/4tc/Dropbox/suli 2018 project/$dirname/"
	mv *.mu.txt "/Users/4tc/Dropbox/suli 2018 project/$dirname/"

	((i++))
done
echo "iteration completed"