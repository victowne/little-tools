import matplotlib.pyplot as plt 
import numpy as np 
import matplotlib as mpl
import matplotlib.tri as tri
import numpy.matlib
from matplotlib import style
import pandas as pd 

  ######## Debug #############
#   import pandas as pd       #
#   df = pd.DataFrame(in_cnt) #
#   df.to_csv('a.txt')        #
  ############################

#print style.available      
plt.style.use('seaborn-paper')
mpl.rcParams['xtick.major.size'] = 0
mpl.rcParams['ytick.major.size'] = 0
#>>>argument
Lmax=3               
Imax=499
m=3
n_angles = 630
#>>>data processing
A = np.loadtxt('xy400.dat')
p = np.loadtxt('Profile0.dat')
x = p[:,0]
eq = p[:,3]
y = np.linspace(0,2.*np.pi,n_angles)
a = A[:,4]
b = A[:,5]
a = a.reshape(Imax+1,Lmax+1,order='F')
b = b.reshape(Imax+1,Lmax+1,order='F')
a = a+1j*b
k = m*np.arange(0,Lmax+1,1)
k = k.reshape(-1,1)      
b = 2*np.exp(1j*k*y)
b[0,:] = b[0,:]/2
pfunc = np.dot(a,b)
pfunc = np.real(pfunc)
pfunc0 = eq+.25*x*x/3
pfunc0 = pfunc0.reshape(-1,1)
pfunc0 = np.matlib.repmat(pfunc0,1,n_angles)
pfunc = pfunc+pfunc0

#>>>cordinate transformation
radii = np.linspace(0., 1., Imax+1)
angles = np.linspace(0, 2*np.pi, n_angles)
angles = np.repeat(angles[..., np.newaxis], Imax+1, axis=1)
#>>>cutoff
pfunc = pfunc[0:250,:]
radii = radii[0:250]
angles = angles[:,0:250]

rad = (radii*np.cos(angles)).flatten()
the = (radii*np.sin(angles)).flatten()
pfunc = pfunc.reshape(-1,1,order='F').flatten()

triang = tri.Triangulation(rad, the)


#>>>plot
plt.tricontourf(triang , pfunc, 20, alpha=.99, cmap='Spectral')
plt.colorbar()
#plt.tricontour(triang, pfunc, 20, colors='black', linewidth=.5)

#>>>axis attribution
#plt.grid()
plt.title(r'$\psi^*$',fontsize=14)
plt.xlabel(r'$r/a$',fontsize=13)
#plt.xlim((0,1))
#plt.xticks(np.linspace(-1, 1, 5))
plt.ylabel(r'$z/a$',fontsize=13)
#plt.ylim((0,1))
#plt.yticks([0, 0.5], ['$minimum$', 'normal'])
#plt.autoscale(tight=True)   

#plt.savefig("C:\\Users\\vic_l\\Desktop\\test.png")
plt.show()