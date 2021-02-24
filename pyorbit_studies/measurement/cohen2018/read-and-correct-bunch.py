dummyBunch = Bunch()
dummyBunch.readBunch("Beam_Correlated.txt",npart)
xtot=xptot=0.0
ytot=yptot=0.0
#Calculate the mean....
for i in range(dummyBunch.getSize()):
	xtot +=dummyBunch.x(i)
	xptot+=dummyBunch.xp(i)
	ytot +=dummyBunch.y(i)
	yptot+=dummyBunch.yp(i)

xMean  =float(xtot)/float(npart)
xpMean =float(xptot)/float(npart)
yMean  =float(ytot)/float(npart)
ypMean =float(yptot)/float(npart)
#Remove mean and rescale.
for i in range(dummyBunch.getSize()):
	x =(dummyBunch.x(i)  - xMean)  * 0.001
	xp=(dummyBunch.xp(i) - xpMean) * 0.001
	y =(dummyBunch.y(i)  - yMean)  * 0.001
	yp=(dummyBunch.yp(i) - ypMean) * 0.001
	b.addParticle(x,xp,y,yp,0.0,0.0)