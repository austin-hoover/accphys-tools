#!/bin/bash

# 
i=1
max=10
s=$(seq -f "%04g" 1 $max)
while [ $i -le $max ]
do
	# 
	itNum=$i
	stepSize=".01"
	echo "rep number $i"
	
	echo "delmu=$stepSize*$itNum ;" >> "delmu.mu.txt"
	/Users/46h/Research/code/madx test-clone.mad
	#rename all the output files
	suf=`printf "%04d" $i`
	dirname="y_optics_$suf"
	mkdir $dirname
	mv *.mu.txt $dirname
	mv *.mu.lat $dirname
	((i++))
done
echo "iteration completed"
