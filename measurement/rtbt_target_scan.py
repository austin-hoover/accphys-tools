"""RTBT target scan simulation.

This script tracks a bunch from the RTBT entrance to the target as the
phase advances at the target are varied. The beam image on the target is
then simulated by smoothing the x-y histogram. This data can be used to 
reconstruct the four-dimensional phase space distribution function using
tomographic methods.

The initial bunch is loaded from a file in '_input/temp/'. Alternatively, a
coasting Gaussian, KV, or Waterbag distribution can be generated from the 
design Twiss parameters or from user-supplied Twiss parameters.
"""

# Setup
#------------------------------------------------------------------------------
