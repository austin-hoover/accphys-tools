"""RTBT wire-scanner emittance measurement simulation.

This script tracks a bunch from the RTBT entrance to the target. Wire-scanner
measurements are then simulated and used to reconstruct the transverse 
covariance matrix. 

The initial bunch is loaded from a file in '_input/temp/'. Alternatively, a
coasting Gaussian, KV, or Waterbag distribution can be generated from the 
design Twiss parameters or from user-supplied Twiss parameters.

The number of wire-scanner measurements, as well as the phase advances at each
measurement, are free parameters. There is also the option to include -0 or 
not include -- the following effects and run a Monte Carlo simulation of the
measurement:
    * wire-scanner noise (very small in reality)
    * wire-scanner tilt angle error
    * energy error
    * quadrupole field errors
    * space charge
    * fringe fields
"""


# Setup
#------------------------------------------------------------------------------
