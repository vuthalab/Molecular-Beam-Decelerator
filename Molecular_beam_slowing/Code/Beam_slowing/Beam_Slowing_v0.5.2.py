#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar  6 10:37:32 2019

@author: Z0RA
"""

#%%
import numpy as np
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint 
from scipy.interpolate import RegularGridInterpolator

# Units
mm = 1e-3
um = 1e-6

# Magnetic field equations
# Functions defining the fields along each axis for the SFS and WFS states
def SFSy(y):
    return (-6267)*y**2 - 0.106*y + 1.018
def SFSx(x):
    return (2.518*10**4)*x**2 - 0.05364*x + 1.021
def WFSy(y):
    return (1.081*10**10)*y**4 + (1.635*10**5)*y**3 - (1.133*10**4)*y**2 - 0.6312*y + 0.02394
def WFSx(x):
    return (7.657*10**9)*x**4 - (1.166*10**5)*x**3 + (3.603*10**4)*x**2 + 0.2786*x + 0.03799
# Magnetic field from ZS-paper archived simulations; B field in [Teslas]
def fullfield(x,y,z):
    return SFSx(x)*SFSy(y)*sin(315.3*z/2)**2 + WFSx(x)*WFSy(y)*cos(315.3*z/2)**2


# Creating arrays and meshgrids
xterms = 349
yterms = 350
zterms = 500
radius = 2.5 # mm
zlen = 20 # mm

# Bore radius is actually x,y = 2.5mm & z = 40mm
x = np.linspace(-radius, radius, xterms)*mm
y = np.linspace(-radius, radius, yterms)*mm
z = np.linspace(0,zlen, zterms)*mm

# When the meshgrid is symmetric, ie size(x) = size(y), they can not all be named X,Y,Z
# for each meshgrid, this the names are written to be cyclic and contain the dimensions of the grid
XY, YX = np.meshgrid(x, y,indexing='ij', sparse=True)
XZ, ZX = np.meshgrid(x, z,indexing='ij', sparse=True)
YZ, ZY = np.meshgrid(y, z,indexing='ij', sparse=True)
XYZ,YZX,ZXY = np.meshgrid(x,y,z,indexing='ij', sparse=True)

# Suggesting some planes 
ypla = 0*mm
zpla = 0*mm

# Gradient of fullfield at y = yplane(y)
Bdxz,Bdzx = np.gradient(fullfield(XZ,ypla,ZX))
# Gradient of fullfield at z = zplane(z)
Bdxy,Bdyx = np.gradient(fullfield(XY,YX,zpla))
# Gradient of the fullfield(x,y,z) at all points
Bdxyz,Bdyzx,Bdzxy = np.gradient(fullfield(XYZ,YZX,ZXY)*2.28e4)
        # Need to add a scaling factor into this above term to satisfy the condition for the slower

# Interpolate the gradient of the fullfield
Bdxfn = RegularGridInterpolator((x,y,z), Bdxyz,bounds_error=False,fill_value=0)
Bdyfn = RegularGridInterpolator((x,y,z), Bdyzx,bounds_error=False,fill_value=0)
Bdzfn = RegularGridInterpolator((x,y,z), Bdzxy,bounds_error=False,fill_value=0)

#%%
# Gradient should be approximately 150T/m to produce a 1 T field, thus to scale to 150T/m,
# divide by maximum gradient of z and then multiply to 150

# Need to find a better way to fix this gradient/B field scaling issue
scale_factor = 150/Bdzfn([0,0,5*mm])
scale_factor # this falue must be approximately == 1 so that the scaling is correct

#%%
zarr = np.arange(0,zlen,0.05)*mm

Bdzarr =[]
for z in np.arange(0,zlen,0.05):
    Bdzarr.append(Bdzfn([0,0,z*mm])[0])

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(zarr, Bdzarr)
ax.set_title('GradB_z vs z')
ax.set_xlabel("z [m]")
ax.set_ylabel("dB/dz [T/m]")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(zarr, fullfield(0,0,zarr))
ax.set_title('B_z vs z')
ax.set_xlabel("z [m]")
ax.set_ylabel("B [m]")

#%%
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XY, YX, fullfield(XY,YX,0),cmap='coolwarm')
ax.set_title('surface')
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("|B| [T]")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XY, YX, fullfield(XY,YX,10*mm),cmap='coolwarm')
ax.set_title('surface')
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("|B| [T]")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XZ, ZX, fullfield(XZ,0,ZX),cmap='coolwarm')
ax.set_title('surface')
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_zlabel("|B| [T]")

#
#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(YZ, ZY, fullfield(0,YZ,ZY),cmap='coolwarm')
#ax.set_title('surface')
#ax.set_xlabel("y [m]")
#ax.set_ylabel("z [m]")
#ax.set_zlabel("|B| [T]")

#%%
# Physical quantities
amu = 1.660539040*10**-27 # amu to kg; NIST
hbar = 1.054571800*10**-34 #J s; NIST
e = 1.6021766208*10**-19 # C; NIST
kb = 1.38064852*10**-23 # J/K

me = 9.10938356*10**-31 # kg; NIST
mYb = 173.045 # amu, for 174Yb, which has an abundance of 31.896%; CIAAW 2015
mO = 15.999 # amu; google
mH =1.00794 #amu; google

# output random values proportional to a gaussian distribution
def randG(mean,std):
    return np.random.normal(mean, std)
# Standard deviation of velocity in x and y comp; is proportional to 3K by design, where
def sigma_v(T):
    return sqrt(kb*T/m)
# change in kinetic energy
def KE(vf,v0):
    dKE = (1/2)*m*((vf)**2-(v0)**2)
    return dKE

#
m_l = 1 # ASSUMPTION, CHECK THIS VALUE

# Properties of YbOH molecule
mYbOH = (mYb+mO+mH)
m = mYbOH * amu
mu_bohr = (-e*hbar)/(2*me) # J/T


# Let mu.B > 0 = WFS state
alpWFS = mu_bohr/m # * grad B = accel
# Let mu.B < 0 = SFS state
alpSFS = -mu_bohr/m # * grad B = accel
#%% Molecule state function, (position dependent)

#Tolerance, if values are below this range, we need to do a calculation for the state of the mol
tol = 1e-6 # mm
init_state = alpWFS # Initial state of molecule

# List of z positions which we need to do a calculation where the molecule may flip
z0_list = np.array([10,20,30,40,50,60,70,80,90,100])*mm
    # Excluding initial point so that we always obtain the initial state defined

global mu_sign

def mol_state(x,y,z):
    global mu_sign
    if (abs(x) < 2.5*mm) and (abs(x) < 2.5*mm) and (z < 100*mm):
        if z == 0: # used to set a fixed initial particle state
            mu_sign = c_s = init_state
        elif any(abs(z0_list - z) < tol):
            if np.random.uniform() < 0.5:
                mu_sign = c_s = alpWFS
            else:
                mu_sign = c_s = alpSFS
        else:
            mu_sign = c_s = mu_sign
    else:
        mu_sign = c_s = 0
    return c_s
#%%
    
tlast = 3500*um
t_terms = 1e5
t = np.linspace(0,tlast,t_terms) # s

global mu_sign

def equation_system(b,t):
    xt,yt,zt,vxt,vyt,vzt = b
    db_over_dt = [vxt,vyt,vzt, mol_state(xt,yt,zt)*Bdxfn([xt,yt,zt%(20*mm)]), mol_state(xt,yt,zt)*Bdyfn([xt,yt,zt%(20*mm)]), mol_state(xt,yt,zt)*Bdzfn([xt,yt,zt%(20*mm)])]
    return db_over_dt


# Initial conditions of the 3 position vectors, and 3 velocity vectors, respectivly
#r0 = np.array([randG(0,3*mm) ,randG(0,3*mm) ,0])*mm     # m 
#v0 = np.array([randG(0,sigma_v(3)), randG(0,sigma_v(3)) ,randG(30,25)])
#r0 = np.array([randG(0,3*mm) ,randG(0,3*mm) ,0])*mm     # m 
#v0 = np.array([randG(0,sigma_v(3)), randG(0,sigma_v(3)) ,30])
r0 = np.array([0,0,0])*mm     # m 
v0 = np.array([0,0,30])  

s0 = np.concatenate([r0,v0])

solution = odeint(equation_system,s0,t)
xt = solution[:,0]
yt = solution[:,1]
zt = solution[:,2]
vxt = solution[:,3]
vyt = solution[:,4]
vzt = solution[:,5]


fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_title('$x - y$ plot')
ax.plot(xt,yt)
ax.set_xlabel(r"$x$ [m]")
ax.set_ylabel(r"$y$ [m]")

# trajectory plot
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('$\Delta KE_{V_z}$ vs z')
ax3.plot(zt,KE(vzt,v0[2])/kb)
ax3.set_xlabel("z [m]")
ax3.set_ylabel("$KE$ [K]")

# Trajectory plot
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_title('Particle Trajectory')
ax4.plot(xt,yt,zt)
ax4.set_xlabel("x [m]")
ax4.set_ylabel("y [m]")
ax4.set_zlabel("z [m]")


#%%

fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_title('$V_x - V_y$ plot')
ax.plot(vxt,vyt)
ax.set_xlabel(r"$v_x$ [m/s]")
ax.set_ylabel(r"$v_y$ [m/s]")

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('Vx vs z')
ax3.plot(zt,vxt)
ax3.set_xlabel("z [m]")
ax3.set_ylabel("Vz [m/s]")

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('Vy vs z')
ax3.plot(zt,vyt)
ax3.set_xlabel("z [m]")
ax3.set_ylabel("Vz [m/s]")

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.grid(True)
ax4.set_title('x vx z')
ax4.plot(zt,xt)
ax4.set_xlabel("z [m]")
ax4.set_ylabel("x [m]")

fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.grid(True)
ax4.set_title('y vs z')
ax4.plot(zt,yt)
ax4.set_xlabel("z [m]")
ax4.set_ylabel("y [m]")
