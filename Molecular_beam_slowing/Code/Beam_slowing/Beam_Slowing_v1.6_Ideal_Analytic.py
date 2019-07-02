
"""
Beam Slowing Version 1.6

Created by:
Henry Vu and James Park
"""
## Packages
import numpy as np
from numba import jit
from numpy import pi,sin,cos,tan,sqrt
import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D
from scipy.integrate import odeint
from scipy.interpolate import RegularGridInterpolator
import cProfile
import threading
import time
import multiprocessing as mp


## Helper Functions

# Magnetic field equations
# Functions defining the fields along each axis for the SFS and WFS states

def SFS_y(y):
    """
    Returns the strong field state in the y-axis

    @type y: float
    @rtype: float
    """
    return (-6267) * y **2 - 0.106 * y + 1.018


def SFS_x(x):
    """
    Returns the strong field state in the x-axis

    @type x: float
    @rtype: float
    """
    return (2.518 * 10**4) * x**2 - 0.05364 * x + 1.021


def WFS_y(y):
    """
    Returns the weak field state in the y-axis

    @type y: float
    @rtype: float
    """
    return (1.081 * 10**10) * y**4 + (1.635*10**5) * y**3 - \
    (1.133 * 10**4) * y**2 - 0.6312 * y + 0.02394


def WFS_x(x):
    """
    Returns the weak field state in the x-axis

    @type x: float
    @rtype: float
    """
    return (7.657 * 10**9) * x**4 - (1.166 * 10**5) * x**3
    + (3.603 * 10**4) * x**2 + 0.2786 * x + 0.03799


def full_field(x, y, z):
    """
    Magnetic field from ZS-paper archived simulations; B field in [Teslas].
    Returns the magnetic field calculation across the x,y,z  plane.

    @type x: float
    @type y: float
    @type z: float
    @rtype: float
    """
    return SFS_x(x) * SFS_y(y) * sin(315.3*z/2)**2 + \
    WFS_x(x) * WFS_y(y) * cos(315.3 * z/2)**2


def rand_gauss(mean, std):
    """
    This function takes a mean value and a standard deviation value and
    output random values proportional to a gaussian distribution.

    @type mean: float
    @type std: float
    @rtype: float
    """
    return np.random.normal(mean, std)


def rand_gauss_trunc(mean, std, cutoff):
    """
    This function is an extension of the rand_gauss function but truncates values
    at a certain cutoff value. If the output is less than the cutoff, repeat this
    until the output is greater than the cutoff.

    @type mean: float
    @type std: float
    @type cutoff: float
    @rtype: float
    """
    val = rand_gauss(mean,std)
    while val <= cutoff:
        val = rand_gauss(mean,std)
    return val


def sigma_v(T):
    """
    Returns the standard deviation of the velocity in the x and y component.
    Is proportional to 3K by design.

    @type T: float
    @rtype: float
    """
    return sqrt(kb*T/m)


def kinetic_energy(vf, v0):
    """
    Returns the change in kinetic energy by the given intial velocity and final
    velocity.

    @type vf: float
    @type v0: float
    @rtype: float
    """
    dKE = (1/2) * m * ((vf)**2 - (v0)**2)
    return dKE

def mol_state(x, y, z):
    """
    Determines the molecular state of the molecule based on its x, y, z
    coordinate. Changes the state from either wfs to sfs or vice versa.

    @type x: float
    @type y: float
    @type z: float
    @rtype: float
    """
   #if (abs(x) < radius*mm) and (abs(y) < radius*mm) and (z < 100*mm):
    if (z < 120*mm):
        if (0 <= z < 10*mm) \
        or (20*mm <= z < 30*mm) \
        or (40*mm <= z < 50*mm) \
        or (60*mm <= z < 70*mm) \
        or (80*mm <= z < 90*mm) \
        or (100*mm <= z < 110*mm):
            current_state = init_state
        else:
            current_state = -init_state #flips the state
    else:
        current_state = 0
    return current_state


def equation_system(b, time_linspace):
    """
    The ordinary system of equations that will be solved in order to determine
    the position of the molecule based on the b-field.

    @type b: float (bfield)
    @type t: Numpy linspace
    @type: List (change of bfield over time)
    """
    xt, yt, zt, vxt, vyt, vzt = b
    db_over_dt = [vxt,vyt,vzt, B_str * mol_state(xt,yt,zt) * Bdxfn([xt,yt,zt%(20*mm)]),
    B_str * mol_state(xt, yt, zt) * Bdyfn([xt, yt, zt % (20 * mm)]),
    B_str * mol_state(xt, yt, zt) * Bdzfn([xt, yt, zt % (20 * mm)])]
    return db_over_dt


def solve_equations(s0):
    """
    Solves the equation of differential equations based on the initial conditions.

    time_linspace is a global variable that is a numpy linspace.

    @type s0: numpy array
    @rtype: numpy array
    """
    solution = odeint(equation_system, s0, time_linspace)
    return solution

## Constants and Terms

# Units
mm = ms = 1e-3
um = us = 1e-6

# Physical quantities
amu = 1.660539040*10**-27 # amu to Kg; NIST
hbar = 1.054571800*10**-34 #J s; NIST
e = 1.6021766208*10**-19 # C; NIST
kb = 1.38064852*10**-23 # J/K
me = 9.10938356*10**-31 # kg; NIST
mYb = 173.045 # amu, for 174Yb, which has an abundance of 31.896%; CIAAW 2015
mO = 15.999 # amu
mH =1.00794 # amu

# Properties of YbOH molecule
mYbOH = (mYb + mO + mH)
m = mYbOH * amu
mu_bohr = (-e*hbar) / (2*me) # J/T

# Terms and Measurements
xterms = yterms = zterms = 400 # ~400 is a good run time, higher the better.
radius = 2.5 # mm
zlen = 20 # mm

#Terms for time
time_last = 5*ms
time_terms = 1e5 # 10000 measurements of time.
time_linspace = np.linspace(0, time_last, time_terms) # s

# Let mu.B > 0 = WFS state
alpWFS = mu_bohr/m # * grad B = accel
# Let mu.B < 0 = SFS state
alpSFS = -mu_bohr/m # * grad B = accel


## Gridding and Linear Spaces

# Generating Linear Space
x = np.linspace(-radius, radius, xterms)*mm
y = np.linspace(-radius, radius, yterms)*mm
z = np.linspace(0,zlen, zterms)*mm
xstepsize = ystepsize = (x.max() - x.min())/len(x) # Assuming uniform stepsize in x,y
zstepsize = (z.max() - z.min())/len(z)
# Produce a 3D grid
XX, YY, ZZ = np.meshgrid(x, y, z, indexing = 'ij', sparse=True)

# Calculating the gradient from the given B-field
B_dx, B_dy, B_dz = np.gradient(full_field(XX, YY, ZZ), xstepsize, ystepsize, zstepsize)
# Interpolate the gradient of the fullfield (Across x, y, z)
Bdxfn = RegularGridInterpolator((x, y, z), B_dx, bounds_error = False, fill_value = 0)
Bdyfn = RegularGridInterpolator((x, y, z), B_dy, bounds_error = False, fill_value = 0)
Bdzfn = RegularGridInterpolator((x, y, z), B_dz, bounds_error = False, fill_value = 0)


## Calculations

# We set the initial state of molecule
init_state = alpWFS
B_str = 1

#import multiprocessing
#multiprocessing.cpu_count()

xt, yt, zt = [], [], []
vxt, vyt,vzt = [], [], []

if __name__ == "__main__":
    num_particles = mp.cpu_count() * 40 #change the constant to adjust particle count.
    ics = []
    for i in range(num_particles):
        r0 = [rand_gauss(0,0.03*mm), rand_gauss(0,0.03*mm), 0]
        v0 = [rand_gauss(0,0.11456), rand_gauss(0,0.11456), rand_gauss(30,2.5)]
        sum = r0 + v0
        ics.append(sum)
    s0 = np.array(ics)
    print("multiprocessing: \n", end='')
    tstart = time.time()
    p = mp.Pool(mp.cpu_count())
    mp_solutions = p.map(solve_equations, s0)
    tend = time.time()
    tmp = tend - tstart
    print("Total runtime: ", tmp)
    print("Total Particles Created: ", num_particles)

    # multi processing speeds it up about ~4.5 times faster on 6 cores.
    # Took about 61 seconds to compute.
    # serial (single core processing) takes about
    # We can use cloud computing to rent about 8 cores and even more.
    # LOOK IN GPU CODING TO MAKE IT EVEN FASTERRR!!!!!!!

    for sol in mp_solutions:
        xt.append(sol[:,0])
        yt.append(sol[:,1])
        zt.append(sol[:,2])
        vxt.append(sol[:,3])
        vyt.append(sol[:,4])
        vzt.append(sol[:,5])

#%% TRUNCATING PHASE SPACE VECTORS TO BE WITHIN AREA OF THE BORE

xtcut, ytcut, ztcut = [], [], []
vxtcut, vytcut,vztcut = [], [], []

for n in range(num_particles):
    # Vectors such that plotting them will always be within the bore
    xtcut.append(((xt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)
    ytcut.append(((yt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)
    ztcut.append(((zt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)
    vxtcut.append(((vxt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)
    vytcut.append(((vyt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)
    vztcut.append(((vzt[n]*1e9)[zt[n] * 1e9 < (120) * 10**6]) / 1e9)


## PHASE SPACE PLOTS along x and y directions

# Position at initial/final times
pos_initial = 0
pos_final = -1

fig = plt.figure()
ax = fig.add_subplot(111) #1 by 1 by 1 grid.
ax.set_title('$V_x - x$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(xtcut[n][pos_initial]/(3*mm),vxtcut[n][pos_initial]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(xtcut[n][pos_final]/(3*mm),vxtcut[n][pos_final]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
#   ax.plot(xt[n]/(3*mm),vxt[n]/(sigma_v(3)), alpha = 0.4)
#   ax.axhline(y = vxt[n][pos_i]/(sigma_v(3)), color = 'orange', linewidth = 0.75)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)    # position of the bore
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)   # position of the bore
ax.set_xlabel("$x/\sigma_x$ ")
ax.set_ylabel("$Vx/\sigma_{V_x}$ ")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_y - y$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(yt[n][pos_initial]/(3*mm),vyt[n][pos_initial]/(sigma_v(3)), s = 10, color = 'red',alpha = 0.4)
    ax.scatter(yt[n][pos_final]/(3*mm),vyt[n][pos_final]/(sigma_v(3)), s = 10, color = 'blue',alpha = 0.4)
#   ax.plot(yt[n]/(3*mm),vyt[n]/(sigma_v(3)), alpha = 0.4)
#   ax.axhline(y = vyt[n][pos_i]/(sigma_v(3)), color = 'orange', linewidth = 0.75)
ax.axvline(x=(radius/3), color = 'salmon', linewidth = 0.75)
ax.axvline(x=(-radius/3), color = 'salmon', linewidth = 0.75)
ax.set_xlabel("$y/\sigma_y$ ")
ax.set_ylabel("$Vy/\sigma_{V_y}$ ")
plt.show()

## Kinetic Energy vs. Z direction

fig = plt.figure()
ax = fig.add_subplot(111)
ax.grid(True)
ax.set_title('$\Delta KE_{V_z}$ vs z')
for n in range(len(xt)):
    ax.plot(ztcut[n]*1e3, kinetic_energy(vztcut[n], vztcut[n][0]) / kb)
ax.set_xlabel("z [mm]")
ax.set_ylabel("$KE$ [K]")

fig = plt.figure()
ax = fig.add_subplot(111)
ax.set_title('$V_z - z$ phase space')
ax.grid()
for n in range(len(xt)):
    ax.scatter(ztcut[n][pos_initial] * 1e3,vztcut[n][pos_initial] / (sigma_v(3)), s = 10, color = 'red', alpha = 0.4)
    ax.scatter(ztcut[n][pos_final] * 1e3,vztcut[n][pos_final] / (sigma_v(3)), s = 10, color = 'blue', alpha = 0.4)
    ax.plot(ztcut[n] * 1e3,vztcut[n] / (sigma_v(3)), alpha = 0.4)
ax.set_xlabel("$z$ [mm]")
ax.set_ylabel("$Vz/\sigma_{V_z}$ ")
plt.show()

## HISTOGRAM of Vz Distribution

vz_initial, vz_final = [], []
for n in range(len(xt)):
    vz_initial.append(vzt[n][0])

for n in range(len(xt)):
    vz_final.append(vzt[n][-1])

fig,ax = plt.subplots()
ax.set_title('Initial and final $V_z$ distrubution')
plt.hist(vz_initial,bins=20, normed=True, alpha = 0.4)
plt.hist(vz_final,bins=20, normed=True, alpha = 0.4)
plt.xlim(0,60)
ax.set_xlabel("$Vz$ [m/s]")
ax.set_ylabel("Normalized counts")
plt.show()
