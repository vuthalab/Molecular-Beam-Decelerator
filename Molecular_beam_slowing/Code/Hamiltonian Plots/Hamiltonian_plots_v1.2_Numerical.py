"""
Created on Sat Apr 27 11:15:12 2019
written by Henry Vu, James Park
"""

# Change directory to where the files are located.



## Python Packages
import os
import numpy as np
from numpy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
from sympy.physics.wigner import wigner_3j,wigner_6j
import sympy as sy
from copy import deepcopy
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import RegularGridInterpolator
import time
os.chdir("/home/james/Desktop/Molecular-Beam-Decelerator/Molecular_beam_slowing/Code/Magnetic Field Plot/")
from textfile_functions import *


## Magnetic Field Equations

# Functions defining the fields along each axis for the SFS and WFS states
def SFSy(y):
    """
    Returns the strong field state along the y-direction.

    @type x: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return (-6267) * y**2 - 0.106 * y + 1.018


def SFSx(x):
    """
    Returns the strong field state along the x-direction.

    @type x: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return (2.518 * 10**4) * x**2 - 0.05364 * x + 1.021


def WFSy(y):
    """
    Returns the weak field state along the y-direction.

    @type x: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return (1.081 * 10**10) * y**4 + (1.635 * 10**5) * y**3 \
    - (1.133 * 10**4) * y**2 - 0.6312 * y + 0.02394


def WFSx(x):
    """
    Returns the weak field state along the x-direction.

    @type x: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return (7.657*10**9) * x**4 - (1.166*10**5) * x**3 \
    + (3.603*10**4) * x**2 + 0.2786 * x + 0.03799


def zfield_sin(z, stagescale):
    """
    Returns a sinusodial B-field along the Z axis.

    @type z: float
    @type stagescale: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return sin(315.3 * z / 2 / stagescale)


def zfield_cos(z, stagescale):
    """
    Returns a cos B-field along the Z axis.

    @type z: float
    @type stagescale: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return cos(315.3 * z / 2 / stagescale)


def fullfield(x, y, z, stagescale):
    """
    Generates the full magnetic field through calculations from the SFS and WFS
    in the x, y direction.

    @type x: float
    @type y: float
    @type z: float
    @type stagescale: float
    @rtype: float (Magnetic Field Magnitude in Tesla)
    """
    return (SFSx(x) * SFSy(y) * zfield_sin(z, stagescale)**2 + WFSx(x) \
    * WFSy(y) * zfield_cos(z,stagescale)**2) * bz_scale + bz_mag_offset


def delta(i, j):
    """
    Delta function, returns 1 if i==j else 0.

    @type i, j: int
    @rtype: int
    """
    if i == j:
        return 1
    else:
        return 0


def curve_inner_points(curve_points, j):
    """
    Obtains all the energy values for one specific curve from the entire
    Hamiltonian matrix.

    @type values: Numpy Array
    @rtype: Numpy Array
    """
    list = []
    for i in range(len(curve_points)):
        list.append(curve_points[i][j])
    return np.array(list)


def gs_numerical_hamiltonian_along_z(axis, bfield_along_z):
    """
    This hamiltonian calculation is for the ground state.

    Creates a 3-D array containing the points that were operated on by the
    hamiltonian. The bfields values are specific values for a discrete z-value
    encompassing the XY-YX meshgrid.

     This new array is used as the surface plot for the hamiltonian.

    @type axis: 2D numpy array
    @type bfield_along_z: 2D numpy array
    @rtype: 3D numpy array
    """
    surface_plots = []
    for i in range(np.size(axis)):
        surface_plots.append([])
    for i in range(len(surface_plots)):
        for j in range(len(surface_plots)):
            surface_plots[i].append(list(lg.eigh(H0 + B_scale * \
        bfield_along_z[i][j] * H_int)[0]))
    surface_plots = np.array(surface_plots)
    return surface_plots


def es_numerical_hamiltonian_along_z(axis, bfield_along_z):
    """
    This hamiltonian calculation is for the excited state.

    Creates a 3-D array containing the points that were operated on by the
    hamiltonian. The bfields values are specific values for a discrete z-value
    encompassing the XY-YX meshgrid.

     This new array is used as the surface plot for the hamiltonian.

    @type axis: 2D numpy array
    @type bfield_along_z: 2D numpy array
    @rtype: 3D numpy array
    """
    surface_plots = []
    for i in range(np.size(axis)):
        surface_plots.append([])
    for i in range(len(surface_plots)):
        for j in range(len(surface_plots)):
            surface_plots[i].append(list(lg.eigh(H0_excited + B_scale * \
        bfield_along_z[i][j] * H_int_excited)[0]))
    surface_plots = np.array(surface_plots)
    return surface_plots


def rearrange_hamiltonian_points(surface_points, k):
    """
    Rearranges the points of hamiltonian matrix into an array of points
    corresponding to 1 Energy surface in a 3 Dimensional Axis.

    @type surface_points: Numpy Array
    @type k: integer
    @rtype: Numpy Array
    """
    list = []
    for i in range(len(surface_points)):
        nested_list = []
        for j in range(len(surface_points[i])):
            nested_list.append(surface_points[i][j][k])
        list.append(nested_list)
    return np.array(list)


def list_to_numpy_matrix(xdim, ydim, bvalues):
    """
    Converts a list of b field values into a xdim*ydim numpy matrix.

    @type xdim: List
    @type ydimL List
    @type bvalues: List
    @rtype: numpy array
    """
    dim_list = []
    for i in range(len(xdim)):
        b_list = []
        for j in range(len(ydim)):
            b_list.append(bvalues.pop(0))
        temp_np = np.array(b_list)
        dim_list.append(temp_np)
    return np.array(dim_list)


## Ground State Hamiltonian equations
def H_rot(A, B):
    '''
    The rotational hamiltonian calculation for the ground state.

    These delta functions all are under the assumption that S (electron and
    I (nucleus) never change, thus we can omit them and only need to worry
    about their particular projections.

    Note, objects A != B

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy value)
    '''
    H_rot = delta(A.N,B.N) * delta(A.mN,B.mN) \
                           * delta(A.mS,B.mS) \
                           * delta(A.mI,B.mI) \
                           * (B_rot * A.N * (A.N+1))
    return H_rot


def H_hfs(A, B):
    """
    Hamiltonian hyperfine structure for the ground state.
    Through the wigner 3-J calculations.

    The two c_hfs terms come about due to the non-spherical, but present
    cylindrical symmetry of the diatomic representation of YbOh.
    If we choose to ignore them, then the energy levels will behave well
    as if it were atomic energy levels.

    Note, objects A != B.

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy Value)
    """
    # hfs J.I
    H_hfs = 0
    for q in (-1,0,1):
        H_hfs += b_hfs * delta(A.N, B.N) * delta(A.mN, B.mN) * delta(A.mF(), B.mF()) \
        * (-1)**q * (-1)**(A.S - A.mS) * wigner_3j(A.S, 1, B.S, -A.mS, q, B.mS) \
        * (-1)**(A.I - A.mI) * wigner_3j(A.I, 1, B.I, -A.mI, -q,B.mI) \
        * np.sqrt(A.S*(A.S+1)*(2*A.S+1)) * np.sqrt(A.I * (A.I+1) * (2 * A.I + 1)) \

        H_hfs += c_hfs * ((delta(A.N,B.N) * delta(A.mN,B.mN) * delta(A.mS,B.mS) \
        * delta(A.mI,B.mI) * (A.mS*A.mI)))

        H_hfs += -c_hfs * ((1/3) * delta(A.N,B.N) * delta(A.mN,B.mN) \
        * delta(A.mF(),B.mF()) * (-1)**q * (-1)**(A.S-A.mS) \
        * wigner_3j(A.S,1,B.S,-A.mS,q,B.mS) * (-1)**(A.I-A.mI) \
        * wigner_3j(A.I,1,B.I,-A.mI,-q,B.mI) * np.sqrt(A.S*(A.S+1) * (2*A.S+1)) \
        * np.sqrt(A.I * (A.I+1) * (2*A.I+1)))
    return H_hfs


def H_mag(A, B):
    """
    Hamiltonian magnetic field.

    Note, objects A != B

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy Value)
    """
    # external B-field S.B
    H_magnetic = delta(A.mI,B.mI) * delta(A.mS,B.mS) * delta(A.mN,B.mN) \
    * delta(A.N,B.N) * (gS*A.mS + gI*A.mI)
    H_magnetic = H_magnetic * -muB
    return H_magnetic


def H_sr(A, B):
    """
    Hamiltonian Spin rotation

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy Value)
    """
    # spin-rotation S.N
    H_sr = 0
    for q in (-1,0,1):
        H_sr += delta(A.mI, B.mI) * delta(A.N, B.N) * delta(A.mF(), B.mF()) * \
        (-1)**q * (-1) ** (A.S - A.mS) * wigner_3j(A.S, 1, B.S, -A.mS, q, B.mS) * \
        (-1)**(A.N - A.mN) * wigner_3j(A.N, 1, B.N, -A.mN, -q, B.mN)

    H_sr = gamma * H_sr * np.sqrt(A.S * (A.S+1) * (2 * A.S+1)) \
    * np.sqrt(A.N * (A.N+1) * (2 * A.N+1))
    return H_sr


## Excited State Hamiltonian Functions

def H_rot_excited(A, B):
    '''
    The rotational hamiltonian calculation for the excited state.

    These delta functions all are under the assumption that S (electron and
    I (nucleus) never change, thus we can omit them and only need to worry
    about their particular projections.

    Note, objects A != B

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy value)
    '''
    # rotational J(J+1)
    H_rot_excited =  delta(A.mI,B.mI) * delta(A.J,B.J) * delta(A.mJ,B.mJ) \
    * (B_rot * A.J * (A.J+1))
    return H_rot_excited


def H_hfs_excited(A, B):
    """
    Hamiltonian hyperfine structure for the excited state.
    Through the wigner 3-J calculations.

    The two c_hfs terms come about due to the non-spherical, but present
    cylindrical symmetry of the diatomic representation of YbOh.
    If we choose to ignore them, then the energy levels will behave well
    as if it were atomic energy levels.

    Note, objects A != B.

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy Value)
    """

    # hfs I.J
    H_hfs_excited = 0
    for q in (-1,0,1):
        H_hfs_excited += b_hfs_excited * delta(A.J,B.J) \
        * delta(A.excited_mF(), B.excited_mF()) * (-1)**q * (-1)**(A.J-A.mJ) \
        * wigner_3j(A.J,1,B.J,-A.mJ,q,B.mJ) * (-1)**(A.I-A.mI) \
        * wigner_3j(A.I,1,B.I,-A.mI,-q,B.mI) * np.sqrt(A.J*(A.J+1)*(2*A.J+1)) \
        * np.sqrt(A.I*(A.I+1)*(2*A.I+1))

        H_hfs_excited += c_hfs_excited \
        * delta(A.J,B.J) * delta(A.mJ,B.mJ) \
        * delta(A.mI,B.mI) * (A.mJ*A.mI)

        H_hfs_excited += -c_hfs_excited * (1/3) * delta(A.J,B.J) \
        * delta(A.excited_mF(), B.excited_mF()) * (-1)**q \
        * (-1)**(A.J-A.mJ) * wigner_3j(A.J,1,B.J,-A.mJ,q,B.mJ) \
        * (-1)**(A.I-A.mI) * wigner_3j(A.I,1,B.I,-A.mI,-q,B.mI) \
        * np.sqrt(A.J*(A.J+1)*(2*A.J+1)) * np.sqrt(A.I*(A.I+1)*(2*A.I+1))
    return H_hfs_excited


def H_mag_excited(A, B):
    """
    Hamiltonian magnetic field for the excited state.

    Note, objects A != B

    @type A: MolecularState Object
    @type B: MolecularState Object
    @rtype: float (Energy Value)
    """
    # external B-field S.B
    H_magnetic_excited = delta(A.mI,B.mI) * delta(A.J,B.J) \
    * delta(A.mJ,B.mJ) * (gJ*A.mJ + gI*A.mI)
    H_magnetic_excited = -muB * H_magnetic_excited
    return H_magnetic_excited


## Class objects
class MolecularState():
    """
    A new class obeject representing a Molecule in a particular orientation
    in it's ground state.
    """
    def __init__(self, N=0, S=1/2, I=1/2):
        """
        @type self: MolecularState Object
        @type N: float (Angular Momentum)
        @rtype S: float (Electron Spin)
        @type I: float (Intrinsic spin)
        """
        self.N = N
        self.mN = N
        self.S = S
        self.mS = S
        self.I = I
        self.mI = I
        self.p = (-1)**self.N

    def __repr__(self):
        """
        Returns a representation of this MolecularState object.

        @type: MolecularState object
        @rtype: String
        """
        attribs = [str(s) for s in [self.N,self.mN,self.S,self.mS,self.I,self.mI]]
        string = ','.join([str(s) for s in attribs])
        return "|" + string + "; " + "mF=" + str(self.mF()) + ">"

    def mF(self):
        """
        What is this?
        """
        return self.mN + self.mS + self.mI


def sublevel_expand(basis):
    """
    Creates all possible permuations from individual elements in our basis.

    @type basis
    @rtype: List[Basis Vectors]
    """
    newbasis = []
    for ket in basis:
        for mN in np.arange(-ket.N,ket.N+1,1):
            for mS in np.arange(-ket.S,ket.S+1,1):
                for mI in np.arange(-ket.I,ket.I+1,1):
                    newket = deepcopy(ket)
                    newket.mN = mN
                    newket.mS = mS
                    newket.mI = mI
                    newbasis.append(newket)
    return newbasis



class ExcitedMolecularState():
    """
    A new class obeject representing a Molecule in a particular orientation
    in it's ground state.
    """
    def __init__(self,J=1/2,I=1/2):
        """
        @type self: ExcitedMolecularState Object
        @type N: float (Angular Momentum)
        @rtype S: float (Electron Spin)
        @type I: float (Intrinsic spin)
        """
        self.J=J
        self.mJ = J
        self.I=I
        self.mI = I

    def __repr__(self):
        """
        Returns a representation of this ExcitedMolecularState object.

        @type: MolecularState object
        @rtype: String
        """
        attribs = [str(s) for s in [self.J,self.mJ,self.I,self.mI]]
        string = ','.join([str(s) for s in attribs])
        return "|" + string + "; " + "mF=" + str(self.excited_mF()) + ">"

    def excited_mF(self):
        """
        What is this?
        """
        return self.mJ + self.mI

    def excited_F(self):
        """
        What is this?
        """
        return self.J + self.I


def excited_sublevel_expand(basis):
    """
    Creates all possible permuations from individual elements in our basis.

    @type basis
    @rtype: List[Basis Vectors]
    """
    newbasis = []
    for ket in basis:
        for mJ in np.arange(-ket.J, ket.J+1, 1):
            for mI in np.arange(-ket.I, ket.I+1, 1):
                newket = deepcopy(ket)
                newket.mJ = mJ
                newket.mI = mI
                newbasis.append(newket)
    return newbasis


class Energy_Curve:
    """
    A energy curve object which will be used to label each energy curve.
    """
    def __init__(self, energy_values, label):
        """
        Initializes a new energy curve with set energy values and a label

        @type self: energy_curve object
        @type var_1: Array
        @type var_2: Array
        @type energy_values: Array
        @type label: String
        @rtype: None
        """
        self.energy_values = energy_values
        self.label = label

    def __repr__(self):
        """
        Returns the string representation of this Energy curve.

        @type self: Energy_curve
        @rtype: string
        """
        return self.label

    def get_energy_values(self):
        """
        Returns the Energy level values in this curve.

        @typse self: Energy_surface
        @rtype: array
        """
        return self.energy_values


## Universal Constants
mm = ms = 1e-3
um = us = 1e-6

stagescale = 20/20
bz_mag_offset = 0.16 # approximately 0.16T if we assume only two stages in the numerical calculation of the Bfield; shift the minimum Bfield value
bz_scale = 0.962 # scale the B field so that the maximum value is approximately 1.16T

B_scale = 1 # Scale the maximum magnitude of the field produced by the analytic Halbach array

# Creating arrays and meshgrids
"""
Change these terms to define the smoothness of the plots.
"""
xterms = 51
yterms = 51
zterms = 41 #change this to 4e3

radius = 10 # mm
zlen = 20 # mm

x = np.linspace(-radius, radius, xterms) * mm
y = np.linspace(-radius, radius, yterms) * mm
z = np.linspace(0, zlen, zterms) * mm

XZ, ZX = np.meshgrid(x, z, indexing='ij', sparse=True)
XY, YX = np.meshgrid(x, y, indexing='ij', sparse=True)

x1 = np.linspace(-radius, radius, xterms) * mm
y1 = np.linspace(-radius, radius, yterms) * mm
z1 = np.linspace(0, zlen, zterms) * mm
e_XZ, e_ZX = np.meshgrid(x1, z1, indexing='ij', sparse=True)


## Ground State Constants
# Molecule constants
B_rot = 7348.4005e-3     # GHz ; molecular rotation constant
gamma= -81.150e-3       # GHz ; spin rotation constant
b_hfs = 4.80e-3  # GHz ; one of the hyperfine constants
c_hfs = 2.46e-3  # GHz ; another hyperfine constant
muB = 14        # GHz/T
gS = -2.0023
gI = 5.585 * 1/1836.152672   # for H nucleus; using same muB for both terms, we divide this term by the difference in magnitude between muB and muN


## Excited States Constants
muB = 14        # GHz/T
gL = -1
gS = -2.0023
gJ = -0.2#-0.002
gI = 5.585 * 1/1836.152672   # for H nucleus; using same muB for both terms, we divide this term by the difference in magnitude between muB and muN

wavenum_freq = 100 * (3 * 10**8) * 1e-9 # convert from cm^-1 to GHz
B_rot_excited = B_rot = 7348.4005e-3     # GHz ; molecular rotation constant
b_hfs_excited = 4.80e-3*gJ  # GHz ; one of the hyperfine constants
c_hfs_excited = 2.46e-3*gJ  # GHz ; another hyperfine constant


## Ground State Hamiltonian Calculations
basis = [MolecularState(N=i, I = 1/2, S = 1/2) for i in range(3)]
# expands the basis to contain all the mN, mI, mS sublevels
basis = sublevel_expand(basis)

N = len(basis)

H0 = np.matrix(np.zeros((N, N))) # Create N x N zero matrix
for i in range(N):
    for j in range(N):
        A, B = basis[j], basis[i]
        H0[j, i] = H_rot(A,B)
        H0[j, i] += H_sr(A,B)
        H0[j, i] += H_hfs(A,B)

H_int = np.matrix(np.zeros((N, N)))
for i in range(N):
    for j in range(N):
        A, B = basis[j], basis[i]
        H_int[j, i] = H_mag(A, B)


## Excited State Hamiltonian Calculations
excited_basis = [ExcitedMolecularState(J=(1/2 + i), I=1/2) for i in range(3)]
#expand the basis to contain all the mN, mI, mS sublevels
excited_basis = excited_sublevel_expand(excited_basis)

N_excited = len(excited_basis)

H0_excited = np.matrix(np.zeros((N_excited,N_excited))) # Create N x N zero matrix
for i in range(N_excited):
    for j in range(N_excited):
        A,B = excited_basis[j], excited_basis[i]
        H0_excited[j,i] = H_rot_excited(A,B)
        H0_excited[j,i] += H_hfs_excited(A,B)

H_int_excited = np.matrix(np.zeros((N_excited, N_excited)))
for i in range(N_excited):
    for j in range(N_excited):
        A,B = excited_basis[j], excited_basis[i]
        H_int_excited[j,i] = H_mag_excited(A,B)


## Loading Bfield values from text-file

#change the directory to the folder containing the b-field data.
os.chdir("/home/james/Desktop/Molecular-Beam-Decelerator/Molecular_beam_slowing/Code/Magnetic Field Plot/Data")
mag_values = load_txtfile_list("bnorm_actual.txt")

# Creates nested list of b-field values.
bfields = [] #change to lst

z_len = []
for i in range(len(z)):
    bfields.append([]) #change to lst
    z_len.append(i)

loop_range = len(mag_values) // len(z)
for i in range(loop_range):
    for i in z_len:
        popped_value = mag_values.pop(0)
        bfields[i].append(popped_value)


# Bfield values at some Z.
start = list_to_numpy_matrix(x, y, bfields[0])
midpoint = list_to_numpy_matrix(x, y, bfields[20])
end = list_to_numpy_matrix(x, y, bfields[40])


## Plotting Hamiltonian Energy levels with Numerical Bfield Values (GROUND STATE)
# BFIELD VALUES FOR XY---YX AT Z = 0
fig = plt.figure("(Ground State) Hamiltonian of Numerical BField at (Z = 0)")
z0_surface_plots_XY = gs_numerical_hamiltonian_along_z(XY, start)
z0_energy_curves = []
for i in range(36):
    #36 refers the the dimension of our Hamiltonian matrix (36x36)
    surface_points_XY = rearrange_hamiltonian_points(z0_surface_plots_XY, i)
    z0_energy_curves.append(Energy_Curve(surface_points_XY, \
    "Ground State Energy Curve " + str(i + 1)))

ax = fig.add_subplot(111, projection='3d')
ax.set_title('(Ground State) Hamiltonian of Numerical BField at (Z = 0)')
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("(Ground State) Energy Levels of Hamiltonian at (Z = 0)")
ax.plot_surface(XY *1e3, YX *1e3, z0_energy_curves[0].energy_values)
ax.plot_surface(XY *1e3, YX *1e3, z0_energy_curves[15].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z0_energy_curves[25].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z0_energy_curves[35].energy_values)
plt.show()


# BFIELD VALUES FOR XY---YX AT Z = 10
fig = plt.figure("(Ground State) Hamiltonian of Numerical BField at (Z = 10)")
z10_surface_plots_XY = gs_numerical_hamiltonian_along_z(XY, midpoint)
z10_energy_curves = []
for i in range(36):
    #36 refers the the dimension of our Hamiltonian matrix (36x36)
    z10_surface_points_XY = rearrange_hamiltonian_points(z10_surface_plots_XY, i)
    z10_energy_curves.append(Energy_Curve(z10_surface_points_XY, \
    "Ground State Energy Curve " + str(i + 1)))
    #ax.plot_surface(XY *1e3, YX *1e3, surface_points_XY)

ax = fig.add_subplot(111, projection='3d')
ax.set_title('(Ground State) Hamiltonian of Numerical BField at (Z = 10)')
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Energy Levels of Hamiltonian at (Z = 10)")
ax.plot_surface(XY *1e3, YX *1e3, z10_energy_curves[0].energy_values)
ax.plot_surface(XY *1e3, YX *1e3, z10_energy_curves[15].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z10_energy_curves[25].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z10_energy_curves[35].energy_values)
plt.show()


## Plotting Hamiltonian Energy levels with Numerical Bfield Values (EXCITED STATE)
# BFIELD VALUES FOR XY---YX AT Z = 0
fig = plt.figure("(Excited_State)Hamiltonian of Numerical BField at (Z = 0)")
z0_surface_plots_XY = es_numerical_hamiltonian_along_z(XY, start)
z0_energy_curves = []
for i in range(24):
    #36 refers the the dimension of our Hamiltonian matrix (36x36)
    surface_points_XY = rearrange_hamiltonian_points(z0_surface_plots_XY, i)
    z0_energy_curves.append(Energy_Curve(surface_points_XY, \
    "Ground State Energy Curve " + str(i + 1)))

ax = fig.add_subplot(111, projection='3d')
ax.set_title('(Excited State) Hamiltonian of Numerical BField at (Z = 0)')
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Energy Levels of Hamiltonian at (Z = 0)")
ax.plot_surface(XY *1e3, YX *1e3, z0_energy_curves[0].energy_values)
ax.plot_surface(XY *1e3, YX *1e3, z0_energy_curves[8].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z0_energy_curves[14].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z0_energy_curves[23].energy_values)
plt.show()


# BFIELD VALUES FOR XY---YX AT Z = 10
fig = plt.figure("(Excited State) Hamiltonian of Numerical BField at (Z = 10)")
z10_surface_plots_XY = es_numerical_hamiltonian_along_z(XY, midpoint)
z10_energy_curves = []
for i in range(24):
    #36 refers the the dimension of our Hamiltonian matrix (36x36)
    z10_surface_points_XY = rearrange_hamiltonian_points(z10_surface_plots_XY, i)
    z10_energy_curves.append(Energy_Curve(z10_surface_points_XY, \
    "Ground State Energy Curve " + str(i + 1)))
    #ax.plot_surface(XY *1e3, YX *1e3, surface_points_XY)

ax = fig.add_subplot(111, projection='3d')
ax.set_title('(Excited State) Hamiltonian of Numerical BField at (Z = 10)')
ax.set_xlabel("X [mm]")
ax.set_ylabel("Y [mm]")
ax.set_zlabel("Energy Levels of Hamiltonian at (Z = 10)")
ax.plot_surface(XY *1e3, YX *1e3, z10_energy_curves[0].energy_values)
ax.plot_surface(XY *1e3, YX *1e3, z10_energy_curves[8].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z10_energy_curves[14].energy_values)
ax.plot_surface(XY *1e3, YX *1e3,  z10_energy_curves[23].energy_values)
plt.show()

