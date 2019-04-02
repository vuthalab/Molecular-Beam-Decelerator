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
bz_mag_offset = 0.16 # approximately 0.16T if we assume only two stages in the numerical calculation of the Bfield; shift the minimum Bfield value
#stagescale == Scale the magnetic field such that two stages extend over the range of the numertor in mm
bz_scale = 0.97 # scale the B field so that the maximum value is approximately 1.16T

def zfield_sin(z,stagescale):
    return sin(315.3*z/2/stagescale)
def zfield_cos(z,stagescale):
    return cos(315.3*z/2/stagescale)

def fullfield(x,y,z,stagescale):
    return (SFSx(x)*SFSy(y)*zfield_sin(z,stagescale)**2 + WFSx(x)*WFSy(y)*zfield_cos(z,stagescale)**2)*bz_scale+bz_mag_offset

def gSFSy(y):
    return -0.106 - 12534*y
def gSFSx(x):
    return -0.05364 + 50360*x
def gWFSy(y):
    return -0.6312 - 22660*y + 490500*y**2 + (4.324*10**10)*y**3
def gWFSx(x):
    return 0.2786 + 72060*x - 349800*x**2 + (3.0628*10**10)*x**3
def gxfullfield(x,y,z,stagescale):
    return (gSFSx(x)*SFSy(y)*zfield_sin(z,stagescale)**2 + gWFSx(x)*WFSy(y)*zfield_cos(z,stagescale)**2)*bz_scale
def gyfullfield(x,y,z,stagescale):
    return (SFSx(x)*gSFSy(y)*zfield_sin(z,stagescale)**2 + WFSx(x)*gWFSy(y)*zfield_cos(z,stagescale)**2)*bz_scale
def gzfullfield(x,y,z,stagescale):
    return (315.3/stagescale)*(SFSx(x)*SFSy(y)*zfield_sin(z,stagescale)*zfield_cos(z,stagescale) - WFSx(x)*WFSy(y)*zfield_cos(z,stagescale)*zfield_sin(z,stagescale))*bz_scale
def gfullfield(x,y,z,stagescale):
    return gxfullfield(x,y,z,stagescale)+gyfullfield(x,y,z,stagescale)+gzfullfield(x,y,z,stagescale)

# Creating arrays and meshgrids
xterms = yterms = zterms = 400 # ~400 is a good run time, but Simulations might be better with a higher number
radius = 2.5 # mm
zlen = 20 # mm

# Bore radius is actually x,y = 2.5mm & z = 40mm
x = np.linspace(-radius, radius, xterms)*mm
y = np.linspace(-radius, radius, yterms)*mm
z = np.linspace(0,zlen, zterms)*mm

#%% PLOT OF THE GRADIENTS OF THE MAGNETIC FIELD

scale = 20/20
zpos = 0*mm

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(z*1e3, gzfullfield(0,0,z,scale))
ax.set_title('GradB_z vs z')
ax.set_xlabel("z [mm]")
ax.set_ylabel("dB/dz [T/m]")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(z*1e3, fullfield(0,0,z,scale))
ax.set_title('B_z vs z')
ax.set_xlabel("z [mm]")
ax.set_ylabel("B [T]")
#%%
fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(x*1e3, gxfullfield(x,0,0,scale))
ax.set_title('GradB_x vs x at z = %.3f mm' %(zpos))
ax.set_xlabel("x [mm]")
ax.set_ylabel("dB/dx [T/m]")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid()
ax.plot(y*1e3, gyfullfield(0,y,0,scale))
ax.set_title('GradB_y vs y at z = %.3f mm' %(zpos))
ax.set_xlabel("y [mm]")
ax.set_ylabel("dB/dy [T/m]")


#%% Physical quantities
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

# Functions for generating gaussian distributions randomly
# output random values proportional to a gaussian distribution
def randG(mean,std):
    return np.random.normal(mean, std)
# If the output is less than cutoff, repeat until output is greater than cutoff
def randG_trunc(mean,std,cutoff):
    val = randG(mean,std)
    while val <= cutoff:
        val = randG(mean,std)
    return val
# Solving for the velocity along one dimension from a temperature T
def velocity(T):
    return sqrt(kb*T/m)
# Difference in kinetic energy equation
def KE(vf,v0):
    dKE = (1/2)*m*((vf)**2-(v0)**2)
    return dKE

#%% IDEAL CASE: Molecule state function
    
# Length of the two stages (where we take the fields centered around a K = 2 Halbach array to cover the length of two total stages (two halfs of K = 6 around the center stage))
len_stage = np.array([20,17,20,13,20,8])

init_state = alpWFS # Initial state of molecule
B_str = 1

def mol_state(x,y,z):
    if (z < np.sum(len_stage)*mm):
        if (0 <= z < len_stage[0]/2*mm) \
        or (len_stage[0]*mm <= z < (len_stage[0]+len_stage[1]/2)*mm) \
        or (np.sum(len_stage[:2])*mm <= z < (np.sum(len_stage[:2])+len_stage[2]/2)*mm) \
        or (np.sum(len_stage[:3])*mm <= z < (np.sum(len_stage[:3])+len_stage[3]/2)*mm) \
        or (np.sum(len_stage[:4])*mm <= z < (np.sum(len_stage[:4])+len_stage[4]/2)*mm) \
        or (np.sum(len_stage[:5])*mm <= z < (np.sum(len_stage[:5])+len_stage[5]/2)*mm):
            c_s = init_state
        else:
            c_s = alpSFS
    else:
        c_s = 0
    return c_s

def stagescale(z):
    if (0 <= z*(1e3) < len_stage[0]):
        val = len_stage[0]/20
    elif (len_stage[0] <= z*(1e3) < np.sum(len_stage[:2])):
        val = len_stage[1]/20
    elif (np.sum(len_stage[:2]) <= z*(1e3) < np.sum(len_stage[:3])):
        val = len_stage[2]/20
    elif (np.sum(len_stage[:3]) <= z*(1e3) < np.sum(len_stage[:4])):
        val = len_stage[3]/20
    elif (np.sum(len_stage[:4]) <= z*(1e3) < np.sum(len_stage[:5])):
        val = len_stage[4]/20
    elif (np.sum(len_stage[:5]) <= z*(1e3) < np.sum(len_stage[:6])):
        val = len_stage[5]/20
    else:
        val = 1 # Unless there is a better assumption to make
    return val

def stagemod(z):
    if (len_stage[0] <= z*(1e3) < np.sum(len_stage[:2])):
        val = np.sum(len_stage[0])*mm
    elif (np.sum(len_stage[:2]) <= z*(1e3) < np.sum(len_stage[:3])):
        val = np.sum(len_stage[:2])*mm
    elif (np.sum(len_stage[:3]) <= z*(1e3) < np.sum(len_stage[:4])):
        val = np.sum(len_stage[:3])*mm
    elif (np.sum(len_stage[:4]) <= z*(1e3) < np.sum(len_stage[:5])):
        val = np.sum(len_stage[:4])*mm
    elif (np.sum(len_stage[:5]) <= z*(1e3) < np.sum(len_stage[:6])):
        val = np.sum(len_stage[:5])*mm
    else:
        val = 1 # Unless there is a better assumption to make
    return val

tlast = 5*ms
t_terms = 1e5
t = np.linspace(0,tlast,t_terms) # s

def equation_system(b,t):
    xt,yt,zt,vxt,vyt,vzt = b
    db_over_dt = [vxt,vyt,vzt, B_str*mol_state(xt,yt,zt)*gxfullfield(xt,yt,zt,1), B_str*mol_state(xt,yt,zt)*gyfullfield(xt,yt,zt,1), B_str*mol_state(xt,yt,zt)*gzfullfield(xt,yt,zt%(stagemod(zt)),stagescale(zt))]
    return db_over_dt

#%%
xt, yt, zt = [], [], []
vxt, vyt,vzt = [], [], []
particle_num = 20
for n in range(particle_num):
    r0 = np.array([randG(0,0.03*mm) ,randG(0,0.03*mm) ,0])# m 
#    v0 = np.array([randG(0,sigma_v(3)), randG(0,sigma_v(3)) ,randG_trunc(30,25,5)])
    v0 = np.array([randG(0,0.11456), randG(0,0.11456) ,randG(30,2.5)])
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
    ax.scatter(xt[n][pos_i]/(3*mm),vxt[n][pos_i]/(velocity(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(xt[n][pos_f]/(3*mm),vxt[n][pos_f]/(velocity(3)), s = 10, color = 'blue',alpha = 0.4)
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
    ax.scatter(yt[n][pos_i]/(3*mm),vyt[n][pos_i]/(velocity(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(yt[n][pos_f]/(3*mm),vyt[n][pos_f]/(velocity(3)), s = 10, color = 'blue',alpha = 0.4)
#    ax.plot(yt[n]/(3*mm),vyt[n]/(sigma_v(3)), alpha = 0.4)
#    ax.axhline(y = vyt[n][pos_i]/(sigma_v(3)), color = 'orange', linewidth = 0.75)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)
ax.set_xlabel("$y/\sigma_y$ ")
ax.set_ylabel("$Vy/\sigma_{V_y}$ ")
#%%
# Position at initial/final times
pos_i = 0
pos_f = -1

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_title('$\Delta KE_{V_z}$ vs z')
for n in range(len(xt)):
    ax.plot(zt[n]*1e3,KE(vzt[n],vzt[n][0])/kb)
for n in range(len(len_stage)):     # Plot vertical lines showing the positions of stages
    ax.axvline(x=np.sum(len_stage[:(n)]))
ax.axvline(x=np.sum(len_stage))
ax.set_xlabel("z [mm]")
ax.set_ylabel("$KE$ [K]")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_z - z$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(zt[n][pos_i]*1e3,vzt[n][pos_i], s = 10, color = 'red',alpha = 0.4)
    ax.scatter(zt[n][pos_f]*1e3,vzt[n][pos_f], s = 10, color = 'blue',alpha = 0.4)
    ax.plot(zt[n]*1e3,vzt[n], alpha = 0.4)
ax.set_xlabel("$z$ [mm]")
ax.set_ylabel("$Vz$ [m/s]")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_z - z$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(zt[n][pos_i]*1e3,(1/2)*m*(vzt[n][pos_i])**2/kb, s = 10, color = 'red',alpha = 0.4)
    ax.scatter(zt[n][pos_f]*1e3,(1/2)*m*(vzt[n][pos_f])**2/kb, s = 10, color = 'blue',alpha = 0.4)
    ax.plot(zt[n]*1e3,(1/2)*m*(vzt[n])**2/kb, alpha = 0.4)
ax.set_xlabel("$z$ [mm]")
ax.set_ylabel("$KE_z$ [K]")
#%%
# HISTOGRAM of Vz Distribution  
vz_initial, vz_final = [], []
for n in range(len(xt)):
    vz_initial.append(vzt[n][0])
    vz_final.append(vzt[n][-1])
fig,ax = plt.subplots()
ax.set_title('Initial and final $V_z$ distrubution')
plt.hist(vz_initial,bins=20,normed=True, alpha = 0.4)
plt.hist(vz_final,bins=20,normed=True, alpha = 0.4)
plt.xlim(0,60)
ax.set_xlabel("$Vz$ [m/s]")
ax.set_ylabel("Normalized counts")
plt.show()

