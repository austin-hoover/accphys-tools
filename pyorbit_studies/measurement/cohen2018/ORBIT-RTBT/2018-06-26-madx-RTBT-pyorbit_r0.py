##############################################################
#
# This script reads the input MAD file with RTBT lattice 
# information, creates the TEAPOT lattice and tracks particles 
# down the RTBT. Writes wirescanner data to file.  
# 06/26/2018 NJE
# 
##############################################################
import sys
import pickle
import math
import random
import numpy as np

from bunch import Bunch

from orbit.bunch_generators import TwissContainer, TwissAnalysis
from orbit.bunch_generators import WaterBagDist2D, GaussDist2D, KVDist2D

# lattice, teapot class
from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import DriftTEAPOT
from orbit.teapot import TEAPOT_MATRIX_Lattice
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from bunch import Bunch, BunchTwissAnalysis

# add injection node, distribution generators 
from orbit.injection import TeapotInjectionNode
from orbit.injection import addTeapotInjectionNode

#from injection import InjectParts
from orbit.injection import UniformLongDist
from orbit.bunch_generators import TwissContainer, TwissAnalysis
from orbit.bunch_generators import KVDist1D, KVDist2D


# add apertures
from orbit.aperture import addTeapotApertureNode
from orbit.aperture import TeapotApertureNode, CircleApertureNode, EllipseApertureNode, RectangleApertureNode
from orbit.aperture import addCircleApertureSet, addEllipseApertureSet, addRectangleApertureSet
from orbit.aperture import Aperture



# trans space charge
from orbit.space_charge.sc2p5d import scAccNodes, scLatticeModifications
from orbit.space_charge.sc3d import scAccNodes, scLatticeModifications
from orbit.space_charge.directforce2p5d import directforceAccNodes, directforceLatticeModifications
from spacecharge import SpaceChargeForceCalc2p5D


from orbit_utils import BunchExtremaCalculator

from orbit.teapot import teapot
from orbit.teapot import TEAPOT_Lattice
from orbit.teapot import DriftTEAPOT
from orbit.lattice import AccLattice, AccNode, AccActionsContainer
from bunch import Bunch, BunchTuneAnalysis
from orbit.utils.orbit_mpi_utils import bunch_orbit_to_pyorbit, bunch_pyorbit_to_orbit
from orbit.kickernodes import XKicker, YKicker
from orbit.kickernodes import rootTWaveform, flatTopWaveform
from orbit.kickernodes import TeapotXKickerNode, TeapotYKickerNode,addTeapotKickerNode
from orbit.diagnostics import StatLats
from orbit.diagnostics import addTeapotDiagnosticsNode
from orbit.diagnostics import TeapotStatLatsNode, TeapotMomentsNode, TeapotTuneAnalysisNode
from orbit.diagnostics import addTeapotStatLatsNodeSet, addTeapotMomentsNodeSet
from orbit.utils import orbitFinalize, NamedObject, ParamsDictObject
from orbit.utils.orbit_mpi_utils import bunch_orbit_to_pyorbit, bunch_pyorbit_to_orbit
from orbit.utils.consts import speed_of_light
from wsNode_diagonal import WS_Node_diagonal

import scipy as sp
from scipy import constants

#############################
#
# Read in the mad-x lattice.
#
#############################

dirInfo = open("/tmp/whereIsLattice.txt", "r")
latticeFile=file.read(dirInfo)
latticeFile=latticeFile.rstrip()
file.close(dirInfo)
# latticeFile = "/Users/4tc/Dropbox/suli 2018 project/optics_0001/latticelist.mu.lat" #changed file name 29/Jun/18
teapot_latt = teapot.TEAPOT_Lattice()
teapot_latt.readMADX(latticeFile,"surv")

# Add the wirescanners
#
#
# These are not the actual positions of the wirescanners
# we need to modify the placement of these to get them right.
# It should be easier now that the markers for the wirescanners are 
# in the mad file.
ws_00 = WS_Node_diagonal("ws_00")
ws_02 = WS_Node_diagonal("ws_02")
ws_20 = WS_Node_diagonal("ws_20")
ws_21 = WS_Node_diagonal("ws_21")
ws_23 = WS_Node_diagonal("ws_23")
ws_24 = WS_Node_diagonal("ws_24")
wscanners = [ws_00, ws_02, ws_20, ws_21, ws_23, ws_24] # Keep track of the scanners for later.

# for i in wscanners:
# 	i.setHistPoints(100,100,100)

(teapot_latt.getNodesForName("ws00")[0]).addChildNode( ws_00, AccNode.EXIT)
(teapot_latt.getNodesForName("ws02")[0]).addChildNode( ws_02, AccNode.EXIT)
(teapot_latt.getNodesForName("ws20")[0]).addChildNode( ws_20, AccNode.EXIT)
(teapot_latt.getNodesForName("ws21")[0]).addChildNode( ws_21, AccNode.EXIT)
(teapot_latt.getNodesForName("ws23")[0]).addChildNode( ws_23, AccNode.EXIT)
(teapot_latt.getNodesForName("ws24")[0]).addChildNode( ws_24, AccNode.EXIT)

#Initialize the lattice - 
teapot_latt.initialize()

length = teapot_latt.getLength()

# Turn off all fringe fields for simplicity.
# We probably don't need to do this. I think this is left over from the file Jeff gave me.  
for node in teapot_latt.getNodes():
    print "node=", node.getName()," type ", node.getType()," s start,stop = %4.3f %4.3f "%teapot_latt.getNodePositionsDict()[node]
    node.setUsageFringeFieldIN(False)
    node.setUsageFringeFieldOUT(False)
    if node.getType() == "quad teapot":
    	print "poles", node.getParam("poles"), "skews", node.getParam("skews"), "kls", node.getParam("kls"), "kq", node.getParam("kq")

# Set up the bunch for tracking.
# Twiss values should come from the values given by Jeff. 
# Check to make sure these are correct - if not change them. 

twissX = TwissContainer(alpha = -8.02501, beta = 23.6752, emittance = 0.00001)
twissY = TwissContainer(alpha = 4.34070, beta = 13.5009, emittance = 0.00001)
#############################
#
# Lattice complete
#
#----------------------------
#
#  Define Distribution
#
#############################

dist   = WaterBagDist2D(twissX,twissY) #We will read in Jeff's distirbutions. 

#Space charge is initially turned off, which we will keep while we test
# out our reconstruction. Once we are sure that the method looks good 
# we can turn on space charge and show how the results change.
macrosize=0. # No space charge!
npart=5000
energy = 1.0 #Gev - BE CAREFUL - We will be using a different energy!
b = Bunch()
b.mass(0.93827231)
b.macroSize(macrosize)
print b.charge()
b.getSyncParticle().kinEnergy(energy)

#------------ For reading distribution from file
# dummyBunch = Bunch()
# dummyBunch.readBunch("/Users/4tc/Dropbox/suli 2018 project/FromJeff/ExtractedBeam/Beam_Correlated.txt",npart)
# xtot=xptot=0.0
# ytot=yptot=0.0
# #Calculate the mean....
# for i in range(dummyBunch.getSize()):
# 	xtot +=dummyBunch.x(i)
# 	xptot+=dummyBunch.xp(i)
# 	ytot +=dummyBunch.y(i)
# 	yptot+=dummyBunch.yp(i)

# xMean  =float(xtot)/float(npart)
# xpMean =float(xptot)/float(npart)
# yMean  =float(ytot)/float(npart)
# ypMean =float(yptot)/float(npart)
# #Remove mean and rescale.
# for i in range(dummyBunch.getSize()):
# 	x =(dummyBunch.x(i)  - xMean)  * 0.001
# 	xp=(dummyBunch.xp(i) - xpMean) * 0.001
# 	y =(dummyBunch.y(i)  - yMean)  * 0.001
# 	yp=(dummyBunch.yp(i) - ypMean) * 0.001
# 	b.addParticle(x,xp,y,yp,0.0,0.0)
#----------------  End reading distribution from file

#------------ For creating distribution from ORBIT (Waterbag)
for i in range(npart):
	(x,xp,y,yp) = dist.getCoordinates()
	b.addParticle(x,xp,y,yp,0.0,0.0)	
#----------------  End creating distribution from ORBIT

bunch_pyorbit_to_orbit(teapot_latt.getLength(), b, "bunch_ORBIT.mu.txt")

addTeapotMomentsNodeSet(teapot_latt, "moments", 3) 
addTeapotStatLatsNodeSet(teapot_latt, "statlats")

print "===========Lattice modified ======================================="
print "New Lattice=",teapot_latt.getName()," length [m] =",teapot_latt.getLength()," nodes=",len(teapot_latt.getNodes())

#print "============= nodes inside the region ==========="
#print all nodes around the specified position
	#for node in teapot_latt.getNodes():
#print "node=",node.getName()," type=",node.getType()," L=",node.getLength()

#=====track bunch ============

b.dumpBunch("initial.mu.txt")
teapot_latt.trackBunch(b)
b.dumpBunch("final.mu.txt")
bunch_pyorbit_to_orbit(teapot_latt.getLength(), b, "bunch2.mu.txt")

print "Writing wirescanner data."
for ws in wscanners:
	print ws.getRMSs()
	ws.writeToASCII()

print "Stop."
