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
mm = ms = 1e-3
um = us = 1e-6

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
xterms = yterms = zterms = 100 # c
radius = 2.5 # mm
zlen = 20 # mm

# Bore radius is actually x,y = 2.5mm & z = 40mm
x = np.linspace(-radius, radius, xterms)*mm
y = np.linspace(-radius, radius, yterms)*mm
z = np.linspace(0,zlen, zterms)*mm

xstepsize = ystepsize = (x.max() - x.min())/len(x) # Assuming uniform stepsize in x,y
zstepsize = (z.max() - z.min())/len(z)

XX,YY,ZZ = np.meshgrid(x,y,z,indexing='ij', sparse=True)

Bdx,Bdy,Bdz = np.gradient(fullfield(XX,YY,ZZ),xstepsize,ystepsize,zstepsize)

# Interpolate the gradient of the fullfield
Bdxfn = RegularGridInterpolator((x,y,z), Bdx,bounds_error=False,fill_value=0)
Bdyfn = RegularGridInterpolator((x,y,z), Bdy,bounds_error=False,fill_value=0)
Bdzfn = RegularGridInterpolator((x,y,z), Bdz,bounds_error=False,fill_value=0)

#%%
zarr = np.arange(0,zlen,0.05)*mm
Bdzarr =[]
for z in np.arange(0,zlen,0.05):
    Bdzarr.append(Bdzfn([0,0,z*mm])[0])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(zarr*1e3, Bdzarr)
ax.set_title('GradB_z vs z')
ax.set_xlabel("z [mm]")
ax.set_ylabel("dB/dz [T/m]")

zpos = 0*mm
xarr = yarr = np.arange(-radius,radius,0.05)*mm # yarr has same dimensions
Bdxarr, Bdyarr = [], []
for x in np.arange(-radius,radius,0.05):
    Bdxarr.append(Bdxfn([x*mm,0,zpos])[0])
    Bdyarr.append(Bdyfn([0,x*mm,zpos])[0])
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(xarr*1e3, Bdxarr)
ax.set_title('GradB_x vs x at z = %.3f mm' %(zpos))
ax.set_xlabel("x [mm]")
ax.set_ylabel("dB/dx [T/m]")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(yarr*1e3, Bdyarr)
ax.set_title('GradB_y vs y at z = %.3f mm' %(zpos))
ax.set_xlabel("y [mm]")
ax.set_ylabel("dB/dy [T/m]")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(zarr*1e3, fullfield(0,0,zarr))
ax.set_title('B_z vs z')
ax.set_xlabel("z [mm]")
ax.set_ylabel("B [T]")

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

# Properties of YbOH molecule
mYbOH = (mYb+mO+mH)
m = mYbOH * amu
mu_bohr = (-e*hbar)/(2*me) # J/T

# Let mu.B > 0 = WFS state
alpWFS = mu_bohr/m # * grad B = accel
# Let mu.B < 0 = SFS state
alpSFS = -mu_bohr/m # * grad B = accel

# output random values proportional to a gaussian distribution
def randG(mean,std):
    return np.random.normal(mean, std)
# If the output is less than cutoff, repeat until output is greater than cutoff
def randG_trunc(mean,std,cutoff):
    val = randG(mean,std)
    while val <= cutoff:
        val = randG(mean,std)
    return val
# Standard deviation of velocity in x and y comp; is proportional to 3K by design, where
def sigma_v(T):
    return sqrt(kb*T/m)
# change in kinetic energy
def KE(vf,v0):
    dKE = (1/2)*m*((vf)**2-(v0)**2)
    return dKE
#%% MOLECULE UNDER STATISTICAL STATE TRANSITIONS

#Tolerance, if values are below this range, we need to do a calculation for the state of the mol
tol = 1e-6 # mm
init_state = alpWFS # Initial state of molecule

# List of z positions which we need to do a calculation where the molecule may flip
z0_list = np.array([10,20,30,40,50,60,70,80,90,100])*mm
    # Excluding initial point so that we always obtain the initial state defined

B_str = 1
global mu_sign
def mol_state(x,y,z):
    global mu_sign
    if (abs(x) < radius*mm) and (abs(y) < radius*mm) and (z < 100*mm):
        if z == 0:
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
    
tlast = 4*ms
t_terms = 1e5
t = np.linspace(0,tlast,t_terms) # s

def equation_system(b,t):
    xt,yt,zt,vxt,vyt,vzt = b
    db_over_dt = [vxt,vyt,vzt, B_str*mol_state(xt,yt,zt)*Bdxfn([xt,yt,zt%(20*mm)]), B_str*mol_state(xt,yt,zt)*Bdyfn([xt,yt,zt%(20*mm)]), B_str*mol_state(xt,yt,zt)*Bdzfn([xt,yt,zt%(20*mm)])]
    return db_over_dt

#%% ONE MOLECULE SIMULATION
    
# Initial conditions of the position r0 and velocity v0
r0 = np.array([0,0,0])     # m 
v0 = np.array([0.4,0.2,30])  

s0 = np.concatenate([r0,v0])

solution = odeint(equation_system,s0,t)
xt = solution[:,0]
yt = solution[:,1]
zt = solution[:,2]
vxt = solution[:,3]
vyt = solution[:,4]
vzt = solution[:,5]

print('\n','Init position ',round(xt[0]*1e6,2),round(yt[0]*1e6,2),round(zt[0]*1e6,2), 'um')
print('Final position ',round(xt[-1]*1e3,2),round(yt[-1]*1e3,2),round(zt[-1]*1e3,2), 'mm')
print(' Init velocity ',round(vxt[0],2),round(vyt[0],2),round(vzt[0],6), 'm/s')
print('Final velocity ',round(vxt[-1],2),round(vyt[-1],2),round(vzt[-1],6), 'm/s')
#print('Time of flight',t[-1])

#fig = plt.figure(figsize=(5,5))
#ax = fig.add_subplot(111)
#ax.grid(True)
#ax.set_title('$x - y$ plot')
#ax.plot(xt,yt)
#ax.set_xlabel(r"$x$ [m]")
#ax.set_ylabel(r"$y$ [m]")

# trajectory plot
fig3 = plt.figure()
ax3 = fig3.add_subplot(111)
ax3.grid(True)
ax3.set_title('$\Delta KE_{V_z}$ vs z')
ax3.plot(zt*1e3,KE(vzt,v0[2])/kb)
ax3.set_xlabel("z [mm]")
ax3.set_ylabel("$KE$ [K]")

## trajectory plot
#fig3 = plt.figure()
#ax3 = fig3.add_subplot(111)
#ax3.grid(True)
#ax3.set_title('$\Delta KE_{V_z}$ vs t')
#ax3.plot(t*1e3,KE(vzt,v0[2])/kb)
#ax3.set_xlabel("t [ms]")
#ax3.set_ylabel("$KE$ [K]")

# Trajectory plot
fig4 = plt.figure()
ax4 = fig4.add_subplot(111, projection='3d')
ax4.set_title('Particle Trajectory')
ax4.plot(xt*1e3,yt*1e3,zt*1e3)
ax4.set_xlabel("x [mm]")
ax4.set_ylabel("y [mm]")
ax4.set_zlabel("z [mm]")
ax4.set_xlim(min(xt)*1e3, max(xt)*1e3)
ax4.set_ylim(min(yt)*1e3, max(yt)*1e3)
ax4.set_zlim(0, max(zt)*1e3)

#%% SIMULATION FOR MULTIPLE PARTICLES

xt, yt, zt = [], [], []
vxt, vyt,vzt = [], [], []
# Leaving these external since code below doesnt always work on the first try

particle_num = 100
for n in range(particle_num):
        # std in velocity is of 3K and rounded to 3 decimals
    r0 = np.array([randG(0,3*mm) ,randG(0,3*mm) ,0])# m 
#    v0 = np.array([randG(0,sigma_v(3)), randG(0,sigma_v(3)) ,randG_trunc(30,25,5)])
    v0 = np.array([randG(0,11.456), randG(0,11.456) ,randG(30,25)])
#    v0 = np.array([randG(0,11.456), randG(0,11.456) ,30])
#    r0 = np.array([0,0,0])*mm     # m 
#    v0 = np.array([0.5,0.2,30])  
    s0 = np.concatenate([r0,v0])
    
    solution = odeint(equation_system,s0,t)
    xt.append(solution[:,0])
    yt.append(solution[:,1])
    zt.append(solution[:,2])
    vxt.append(solution[:,3])
    vyt.append(solution[:,4])
    vzt.append(solution[:,5])
    print(n)

#%% PHASE SPACE PLOTS along x and y directions

# Position at initial/final times
pos_i = 0
pos_f = -1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_x - x$ phase space at $t = t_0$')
ax.grid()
for n in range(len(xt)):
    ax.scatter(xt[n][pos_i]/(3*mm),vxt[n][pos_i]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(xt[n][pos_f]/(3*mm),vxt[n][pos_f]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
#    ax.plot(xt[n]/(3*mm),vxt[n]/(sigma_v(3)), alpha = 0.4)
#    ax.axhline(y = vxt[n][pos_i]/(sigma_v(3)), color = 'orange', linewidth = 0.75)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)    # position of the bore
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)   # position of the bore
ax.set_xlabel("$x/\sigma_x$ ")
ax.set_ylabel("$Vx/\sigma_{V_x}$ ")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_y - y$ phase space at $t = t_0$')
ax.grid()
for n in range(len(xt)):
    ax.scatter(yt[n][pos_i]/(3*mm),vyt[n][pos_i]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(yt[n][pos_f]/(3*mm),vyt[n][pos_f]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
#    ax.plot(yt[n]/(3*mm),vyt[n]/(sigma_v(3)), alpha = 0.4)
#    ax.axhline(y = vyt[n][pos_i]/(sigma_v(3)), color = 'orange', linewidth = 0.75)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)
ax.set_xlabel("$y/\sigma_y$ ")
ax.set_ylabel("$Vy/\sigma_{V_y}$ ")
#
#fig = plt.figure()
#ax = fig.add_subplot(111)
#ax.set_title('$V_z - z$')
#ax.grid()
#for n in range(len(zt)):
#    ax.plot(zt[n],vzt[n], alpha = 0.4)
#ax.set_xlabel("$z ")
#ax.set_ylabel("$Vz$ ")









#%% IDEAL CASE: Molecule state function, (position dependent)

init_state = alpWFS # Initial state of molecule
B_str = 1

def mol_state(x,y,z):
#    if (abs(x) < radius*mm) and (abs(y) < radius*mm) and (z < 100*mm):
    if (z < 100*mm):
        if (0 <= z < 10*mm) or (20*mm <= z < 30*mm) or (40*mm <= z < 50*mm) or (60*mm <= z < 70*mm) or (80*mm <= z < 90*mm):
            c_s = init_state
        else:
            c_s = -init_state
    else:
        c_s = 0
    return c_s
    

tlast = 5*ms
t_terms = 1e5
t = np.linspace(0,tlast,t_terms) # s

def equation_system(b,t):
    xt,yt,zt,vxt,vyt,vzt = b
    db_over_dt = [vxt,vyt,vzt, B_str*mol_state(xt,yt,zt)*Bdxfn([xt,yt,zt%(20*mm)]), B_str*mol_state(xt,yt,zt)*Bdyfn([xt,yt,zt%(20*mm)]), B_str*mol_state(xt,yt,zt)*Bdzfn([xt,yt,zt%(20*mm)])]
    return db_over_dt

#%%
xt, yt, zt = [], [], []
vxt, vyt,vzt = [], [], []
# Leaving these external since code below doesnt always work on the first try

particle_num = 5
for n in range(particle_num):
        # std in velocity is of 3K and rounded to 3 decimals
    r0 = np.array([randG(0,0.03*mm) ,randG(0,0.03*mm) ,0])# m 
#    v0 = np.array([randG(0,sigma_v(3)), randG(0,sigma_v(3)) ,randG_trunc(30,25,5)])
    v0 = np.array([randG(0,0.11456), randG(0,0.11456) ,randG(30,2.5)])
#    v0 = np.array([randG(0,11.456), randG(0,11.456) ,30])
#    r0 = np.array([0,0,0])*mm     # m 
#    v0 = np.array([0.5,0.2,30])  
    s0 = np.concatenate([r0,v0])
    
    solution = odeint(equation_system,s0,t)
    xt.append(solution[:,0])
    yt.append(solution[:,1])
    zt.append(solution[:,2])
    vxt.append(solution[:,3])
    vyt.append(solution[:,4])
    vzt.append(solution[:,5])
    print(n)
    
#%% PHASE SPACE PLOTS along x and y directions

# Position at initial/final times
pos_i = 0
pos_f = -1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_x - x$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(xt[n][pos_i]/(3*mm),vxt[n][pos_i]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(xt[n][pos_f]/(3*mm),vxt[n][pos_f]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
#    ax.plot(xt[n]/(3*mm),vxt[n]/(sigma_v(3)), alpha = 0.4)
#    ax.axhline(y = vxt[n][pos_i]/(sigma_v(3)), color = 'orange', linewidth = 0.75)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)    # position of the bore
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)   # position of the bore
ax.set_xlabel("$x/\sigma_x$ ")
ax.set_ylabel("$Vx/\sigma_{V_x}$ ")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_y - y$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(yt[n][pos_i]/(3*mm),vyt[n][pos_i]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(yt[n][pos_f]/(3*mm),vyt[n][pos_f]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
#    ax.plot(yt[n]/(3*mm),vyt[n]/(sigma_v(3)), alpha = 0.4)
#    ax.axhline(y = vyt[n][pos_i]/(sigma_v(3)), color = 'orange', linewidth = 0.75)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)
ax.set_xlabel("$y/\sigma_y$ ")
ax.set_ylabel("$Vy/\sigma_{V_y}$ ")

#%%

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_title('$\Delta KE_{V_z}$ vs z')
for n in range(len(xt)):
    ax.plot(zt[n]*1e3,KE(vzt[n],vzt[n][0])/kb)
ax.set_xlabel("z [mm]")
ax.set_ylabel("$KE$ [K]")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_z - z$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(zt[n][pos_i]*1e3,vzt[n][pos_i]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(zt[n][pos_f]*1e3,vzt[n][pos_f]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
    ax.plot(zt[n]*1e3,vzt[n]/(sigma_v(3)), alpha = 0.4)
ax.set_xlabel("$z$ [mm]")
ax.set_ylabel("$Vz/\sigma_{V_z}$ ")

#%% HISTOGRAM of Vz Distribution
    
vz_initial, vz_final = [], []
for n in range(len(xt)):
    vz_initial.append(vzt[n][0])

for n in range(len(xt)):
    vz_final.append(vzt[n][-1])
    
fig,ax = plt.subplots()
ax.set_title('Initial and final $V_z$ distrubution')
plt.hist(vz_initial,bins=20,normed=True, alpha = 0.4)
plt.hist(vz_final,bins=20,normed=True, alpha = 0.4)
plt.xlim(0,60)
ax.set_xlabel("$Vz$ [m/s]")
ax.set_ylabel("Normalized counts")
plt.show()

#%% NO B FIELD CASE
    
tlast = 5*ms
t_terms = 1e5
t = np.linspace(0,tlast,t_terms) # s

def equation_system(b,t):
    xt,yt,zt,vxt,vyt,vzt = b
    db_over_dt = [vxt,vyt,vzt, 0, 0, 0]
    return db_over_dt
    
xt, yt, zt = [], [], []
vxt, vyt,vzt = [], [], []
# Leaving these external since code below doesnt always work on the first try

particle_num = 100
for n in range(particle_num):
        # std in velocity is of 3K and rounded to 3 decimals
    r0 = np.array([randG(0,3*mm) ,randG(0,3*mm) ,0])# m 
#    v0 = np.array([randG(0,sigma_v(3)), randG(0,sigma_v(3)) ,randG_trunc(30,25,5)])
    v0 = np.array([randG(0,11.456), randG(0,11.456) ,randG(30,25)])
#    v0 = np.array([randG(0,11.456), randG(0,11.456) ,30])
#    r0 = np.array([0,0,0])*mm     # m 
#    v0 = np.array([0.5,0.2,30])  
    s0 = np.concatenate([r0,v0])
    
    solution = odeint(equation_system,s0,t)
    xt.append(solution[:,0])
    yt.append(solution[:,1])
    zt.append(solution[:,2])
    vxt.append(solution[:,3])
    vyt.append(solution[:,4])
    vzt.append(solution[:,5])
    print(n)
    
#%% PHASE SPACE PLOTS along x and y directions

# Position at initial/final times
pos_i = 0
pos_f = -1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('No Field $V_x - x$ phase space at $t = t_0$')
ax.grid()
for n in range(len(xt)):
    ax.scatter(xt[n][pos_i]/(3*mm),vxt[n][pos_i]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(xt[n][pos_f]/(3*mm),vxt[n][pos_f]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)    # position of the bore
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)   # position of the bore
ax.set_xlabel("$x/\sigma_x$ ")
ax.set_ylabel("$Vx/\sigma_{V_x}$ ")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('No Field $V_y - y$ phase space at $t = t_0$')
ax.grid()
for n in range(len(xt)):
    ax.scatter(yt[n][pos_i]/(3*mm),vyt[n][pos_i]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(yt[n][pos_f]/(3*mm),vyt[n][pos_f]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)
ax.set_xlabel("$y/\sigma_y$ ")
ax.set_ylabel("$Vy/\sigma_{V_y}$ ")
