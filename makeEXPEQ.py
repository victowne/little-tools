import numpy as np
import matplotlib.pyplot as plt
#Specify Plasma Boundary#
#Dee Formula
tdum = np.linspace(-np.pi,np.pi,301);
rmax = 0.75
rzero = 2.0
eshape = 1.0 #elongation
xshape = 0.0 #triangularity
R = rzero + rmax * (np.cos(tdum) - xshape * np.sin(tdum)**2)
Z = eshape * rmax * np.sin(tdum)
plt.scatter(R,Z)
plt.axis('equal')
plt.show()
