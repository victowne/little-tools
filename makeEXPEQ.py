import numpy as np
import matplotlib.pyplot as plt
nbound = 301 #number of (R,Z)
NWBPS = 2 # =1,only plasma; =2 plasma&wall
NDATA = 2 # =2,uniform resistivity
NPPF1 = 201 # grid number of the profiles
NSTTP = 2 # =1,TT';=2,toroidal current density;=3,parallel current density
#Specify Plasma Boundary#
#Dee Formula
rmax = 0.43
rzero = 1.9
eshape = 1.0 #elongation
xshape = 0.0 #triangularity
tdum = np.linspace(-np.pi,np.pi,nbound);
R = rzero + rmax * (np.cos(tdum) - xshape * np.sin(tdum)**2)
Z = eshape * rmax * np.sin(tdum)
rmax = rmax + 0.1
R2 = rzero + rmax * (np.cos(tdum) - xshape * np.sin(tdum)**2)
Z2 = eshape * rmax * np.sin(tdum)
epsilon = rmax/rzero

#Profiles
s = 
