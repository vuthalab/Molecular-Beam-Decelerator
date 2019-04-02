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

# Functions defining the fields along each axis for the SFS and WFS states
def SFSy(y):
    return (-6267)*y**2 - 0.106*y + 1.018
def SFSx(x):
    return (2.518*10**4)*x**2 - 0.05364*x + 1.021
def WFSy(y):
    return (1.081*10**10)*y**4 + (1.635*10**5)*y**3 - (1.133*10**4)*y**2 - 0.6312*y + 0.02394
def WFSx(x):
    return (7.657*10**9)*x**4 - (1.166*10**5)*x**3 + (3.603*10**4)*x**2 + 0.2786*x + 0.03799

# Magnetic fields from ZS-paper archived simulations
# B field [Teslas]
def fullfield(x,y,z):
    return SFSx(x)*SFSy(y)*sin(315.3*z/2)**2 + WFSx(x)*WFSy(y)*cos(315.3*z/2)**2

#%%
# Creating arrays and meshgrids
xterms = 300
yterms = 310
zterms = 420
radius = 2.5 # mm
zlen = 100 # mm

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
Bdxyz,Bdyzx,Bdzxy = np.gradient(fullfield(XYZ,YZX,ZXY)*1e4/4*1.5)


# Interpolate the gradient of the fullfield
Bdxfn = RegularGridInterpolator((x,y,z), Bdxyz,bounds_error=False,fill_value=0)
Bdyfn = RegularGridInterpolator((x,y,z), Bdyzx,bounds_error=False,fill_value=0)
Bdzfn = RegularGridInterpolator((x,y,z), Bdzxy,bounds_error=False,fill_value=0)

#%%
zarr = np.arange(0,30,0.05)*mm

Bdzarr =[]
for z in np.arange(0,30,0.05):
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
ax.set_title('GradB_z vs z')
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

# COMPARING DIFFERENCE BETWEEN NORMAL or EXTENDED X, Y, Z RANGES

x = np.linspace(-5, 5, 200)*mm
y = np.linspace(-5, 5, 210)*mm
z = np.linspace(0,60, 420)*mm

# When the meshgrid is symmetric, ie size(x) = size(y), they can not all be named X,Y,Z
# for each meshgrid, this the names are written to be cyclic and contain the dimensions of the grid
XY2, YX2 = np.meshgrid(x, y,indexing='ij', sparse=True)
XZ2, ZX2 = np.meshgrid(x, z,indexing='ij', sparse=True)
YZ2, ZY2 = np.meshgrid(y, z,indexing='ij', sparse=True)
XYZ2,YZX2,ZXY2 = np.meshgrid(x,y,z,indexing='ij', sparse=True)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XY2, YX2, fullfield(XY2,YX2,0),cmap='coolwarm')
ax.set_title('surface')
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("|B| [T]")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XY2, YX2, fullfield(XY2,YX2,10*mm),cmap='coolwarm')
ax.set_title('surface')
ax.set_xlabel("x [m]")
ax.set_ylabel("y [m]")
ax.set_zlabel("|B| [T]")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XZ2, ZX2, fullfield(XZ2,0,ZX2),cmap='coolwarm')
ax.set_title('surface')
ax.set_xlabel("x [m]")
ax.set_ylabel("z [m]")
ax.set_zlabel("|B| [T]")

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(XY2, YX2, fullfield(XY2,YX2,10*mm),cmap='coolwarm')
ax.plot_surface(XY, YX, fullfield(XY,YX,10*mm)-0.1,cmap='viridis')
#ax.set_zlim3d(0,2)

# OVERLAP OF THESE FUNCTION SUGGEST THAT YOU CAN JUST EXTEND THE RANGE OF THE MESH GRID AND THE
# FIELD STRENGTHS ARE THE SAME
fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(XY2, fullfield(XY2,0,0*mm))
ax.plot(XY, fullfield(XY,0,0*mm))

fig = plt.figure()
ax = fig.add_subplot(111)
ax.plot(XY2, fullfield(XY2,0,10*mm))
ax.plot(XY, fullfield(XY,0,10*mm))

#%%
# Plot a cylinder

x=np.linspace(-1, 1, 100)
z=np.linspace(-2, 2, 100)
Xc, Zc=np.meshgrid(x, z)
Yc = np.sqrt(1-Xc**2)

def data_for_cylinder_along_z(center_x,center_y,radius,height_z):
    z = np.linspace(0, height_z, 50)
    theta = np.linspace(0, 2*np.pi, 50)
    theta_grid, z_grid=np.meshgrid(theta, z)
    x_grid = radius*np.cos(theta_grid) + center_x
    y_grid = radius*np.sin(theta_grid) + center_y
    return x_grid,y_grid,z_grid


Xc,Yc,Zc = data_for_cylinder_along_z(0,0,2.5*mm,zlen*mm)


#fig = plt.figure()
#ax = fig.add_subplot(111, projection='3d')
#ax.plot_surface(Xc, Yc, Zc, alpha=0.4)

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

# The B field maxes out at z = 10, 30 mm, 
# Thus we want to transition from WFS --> SFS at those points
# BUT also want to go from SFS --> WFS at z = 20mm, where the field B is zero

# Function to produce the correct sign of alpha depending on the position
#   differentiats between SFS and WFS states


def KE(vf,v0):
    dKE = (1/2)*m*((vf)**2-(v0)**2)
    return dKE

# Random strength parameter
Bstr = 1 # Strength of the fullfield is max around 1 T by design
#%%
def alp(xp,yp,zp):
    if (abs(xp) <= 2.5*mm) and (abs(yp) <= 2.5*mm) and (zp >= 10*mm and zp < 20*mm) or (zp >= 30*mm and zp <= 40*mm):
        alpha = alpSFS
    else:
        alpha = alpWFS
    return alpha*Bstr

#%%
    
# Define random state once molecule reaches certain positions and maintains this state until another
# position n=1,2,3,... away

def alp(xp,yp,zp, c_s):
    # Initial state
    #alpha = alpWFS
    if (abs(xp) <= 2.5*mm) and (abs(yp) <= 2.5*mm) and (zp <= 100*mm):
        if zp %10*mm == 0:
            if np.random.uniform() < 0.5:
                c_s = alpWFS
            else:
                c_s = alpSFS
            print(c_s)
            return c_s
        else:
            print(c_s)
            return c_s
            

#%%
mean_x, sigma_x = 0, 3*mm
x0 = np.random.normal(mean_x, sigma_x,1000)

# Create the bins and histogram
count, bins, ignored = plt.hist(x0, 20, normed=True)
# Plot the distribution curve
plt.plot(bins, 1/(sigma_x * np.sqrt(2 * np.pi))*np.exp( - (bins - mean_x)**2 / (2 * sigma_x**2) ),linewidth=3, color='y')
plt.show() 
#%%

# output random values proportional to a gaussian distribution
def randG(mean,std):
    return np.random.normal(mean, std)

# std of velocity in x and y comp is proportional to 3K, where
def sigma_v(T):
    return sqrt(kb*T/m)

#%%

tlast = 6.0*mm
tlength = 1e5
t = np.linspace(0,tlast,tlength) # ms 


c_s = alpWFS
def equation_system(b,t):
    xt,yt,zt,vxt,vyt,vzt = b
    if zt == 0:
        c_s = alpWFS
    else:
        #zt = int(zt*100000)/100000
        print(zt)
        if zt%10*mm == 0:
            if np.random.uniform() < 0.5:
                c_s = alpWFS
            else:
                c_s = alpSFS
        else:
            c_s = alpWFS
        
    db_over_dt = [vxt,vyt,vzt, c_s*Bdxfn([xt,yt,zt]) , c_s*Bdyfn([xt,yt,zt]), c_s*Bdzfn([xt,yt,zt])]
    #db_over_dt = [vxt,vyt,vzt, alpWFS*Bdxfn([xt,yt,zt]) , alpWFS*Bdyfn([xt,yt,zt]), alpWFS*Bdzfn([xt,yt,zt])]
    #db_over_dt = [vxt,vyt,vzt, alpSFS*Bdxfn([xt,yt,zt]) , alpSFS*Bdyfn([xt,yt,zt]), alpSFS*Bdzfn([xt,yt,zt])]
    return db_over_dt

# Initial conditions of the 3 position vectors, and 3 velocity vectors, respectivly
r0 = np.array([randG(0,3*mm) ,randG(0,3*mm) ,0])*mm     # m 
v0 = np.array([randG(0,sigma_v(0.001)), randG(0,sigma_v(0.001)) ,randG(30,25)])
s0 = np.concatenate([r0,v0])


#solution = solve_ivp(equation_system, [0, 50], s0)
    
solution = odeint(equation_system,s0,t)
odeint

xt = solution[:,0]
yt = solution[:,1]
zt = solution[:,2]
vxt = solution[:,3]
vyt = solution[:,4]
vzt = solution[:,5]

#%%
# plot
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_title('vz vs t')
ax.plot(t,vzt-v0[2])
ax.set_xlabel("t [s]")
#ax.set_xlim(0,tlast)
ax.set_ylabel("vz [m/s]")

'''
# trajectory plot
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('vz vs z')
ax3.plot(zt,vzt-v0[2])
ax3.set_xlabel("z [m]")
ax3.set_ylabel("vz [m/s]")
'''

# trajectory plot
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('$\Delta KE_{V_z}$ vs t')
ax3.plot(t,KE(vzt,v0[2])/kb)
ax3.set_xlabel("t [s]")
ax3.set_ylabel("$KE$ [K]")

# trajectory plot
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('$\Delta KE_{V_z}$ vs z')
ax3.plot(zt,KE(vzt,v0[2])/kb)
ax3.set_xlabel("z [s]")
ax3.set_ylabel("$KE$ [K]")

'''
# trajectory plot
fig2 = plt.figure()
ax2 = fig2.add_subplot(111)
ax2.grid(True)
ax2.set_title('z vs t')
ax2.plot(t,zt)
ax2.set_xlabel("t [s]")
ax2.set_ylabel("z [m]")
'''

# Trajectory plot
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_title('Particle Trajectory')
ax4.plot(xt,yt,zt)
ax4.set_xlabel("x [m]")
ax4.set_ylabel("y [m]")
ax4.set_zlabel("z [m]")


#%%

# vx-vy plot
fig = plt.figure(figsize=(5,5))
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_title('$V_x - V_y$ plot')
ax.plot(vxt,vyt)
ax.set_xlabel(r"$v_x$ [m/s]")
ax.set_ylabel(r"$v_y$ [m/s]")

# Velocity plots
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('Vx velocity vs time')
ax3.plot(t,vxt)
ax3.set_xlabel("time [s]")
ax3.set_ylabel("Vx [m/s]")

fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('Vy velocity vs time')
ax3.plot(t,vyt)
ax3.set_xlabel("time [s]")
ax3.set_ylabel("Vy [m/s]")


# trajectory plot
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('Vx vs z')
ax3.plot(zt,vxt)
ax3.set_xlabel("Z [m]")
ax3.set_ylabel("Vz [m/s]")

# trajectory plot
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('Vy vs z')
ax3.plot(zt,vyt)
ax3.set_xlabel("Z [m]")
ax3.set_ylabel("Vz [m/s]")

# Trajectory plot
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.grid(True)
ax4.set_title('Particle Trajectory')
ax4.plot(zt,xt)
ax4.set_xlabel("z [m]")
ax4.set_ylabel("x [m]")

# Trajectory plot
fig4 = plt.figure()
ax4 = fig4.add_subplot(111)
ax4.grid(True)
ax4.set_title('Particle Trajectory')
ax4.plot(zt,yt)
ax4.set_xlabel("z [m]")
ax4.set_ylabel("y [m]")