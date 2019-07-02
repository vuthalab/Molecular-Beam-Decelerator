#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Apr 27 11:15:12 2019
written by Henry Vu

Molecular states to produce energy spectrum of ground doublet Sigma+ state of YbOH.
Basis for excited state spectrum is J,mJ,I,mI in (ver 1.2.2)
"""

#%% Hyperfine structure of YbOH
import numpy as np
from numpy import linalg as lg
from numpy import pi,sin,cos,tan,sqrt
from sympy.physics.wigner import wigner_3j,wigner_6j
import sympy as sy
from copy import deepcopy
import matplotlib.pyplot as plt

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
def zfield_sin(z,stagescale):
    return sin(315.3*z/2/stagescale)
def zfield_cos(z,stagescale):
    return cos(315.3*z/2/stagescale)
def fullfield(x,y,z,stagescale):
    return (SFSx(x)*SFSy(y)*zfield_sin(z,stagescale)**2 + WFSx(x)*WFSy(y)*zfield_cos(z,stagescale)**2)*bz_scale+bz_mag_offset

stagescale = 20/20
bz_mag_offset = 0.16 # approximately 0.16T if we assume only two stages in the numerical calculation of the Bfield; shift the minimum Bfield value
bz_scale = 0.962 # scale the B field so that the maximum value is approximately 1.16T

B_scale = 1 # Scale the maximum magnitude of the field produced by the analytic Halbach array

zterms = 4e3
zlen = 20 # mm
z = np.linspace(0,zlen, zterms)*mm
#%%
class MolecularState():
    def __init__(self,N=0,S=1/2,I=1/2):
        self.N=N
        self.mN = N
        self.S=S
        self.mS = S
        self.I=I
        self.mI = I
        self.p = (-1)**self.N
        
    def __repr__(self):
        attribs = [str(s) for s in [self.N,self.mN,self.S,self.mS,self.I,self.mI]]
        string = ','.join([str(s) for s in attribs])
        return "|" + string + "; " + "mF=" + str(self.mF()) + ">"
    
    def mF(self):
        return self.mN + self.mS + self.mI
    
    def F(self):
        return self.N + self.S + self.I
    
    def J(self):
        return self.N + self.S
    
    def mJ(self):
        return self.mN + self.mS
    
def sublevel_expand(basis):
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

def delta(i,j): 
    if i==j: return 1
    else: return 0

## molecule properties & functions
B_rot = 7348.4005e-3     # GHz ; molecular rotation constant
gamma= -81.150e-3       # GHz ; spin rotation constant
b_hfs = 4.80e-3  # GHz ; one of the hyperfine constants
c_hfs = 2.46e-3  # GHz ; another hyperfine constant
muB = 14        # GHz/T
gS = -2.0023
gI = 5.585 * 1/1836.152672   # for H nucleus; using same muB for both terms, we divide this term by the difference in magnitude between muB and muN


def H_rot(A,B):
    # rotational N(N+1)
    H_rot = delta(A.N,B.N) * delta(A.mN,B.mN) * delta(A.mS,B.mS) * delta(A.mI,B.mI) * (B_rot * A.N * (A.N+1))
    '''
    These delta functions all are under the assumption that S (electron and I (nucleus) never change,
    thus we can omit them and only need to worry about their particular projections
    '''
    return H_rot

def H_sr(A,B):
    # spin-rotation S.N
    H_sr = 0
    for q in (-1,0,1):
        H_sr += delta(A.mI,B.mI) * delta(A.N,B.N) * delta(A.mF(),B.mF()) * (-1)**q * (-1)**(A.S-A.mS) * wigner_3j(A.S,1,B.S,-A.mS,q,B.mS) * (-1)**(A.N-A.mN) * wigner_3j(A.N,1,B.N,-A.mN,-q,B.mN) 
    H_sr = gamma * H_sr * np.sqrt(A.S*(A.S+1)*(2*A.S+1)) * np.sqrt(A.N*(A.N+1)*(2*A.N+1))
    return H_sr

def H_hfs(A,B):
    # hfs J.I
    H_hfs = 0
    for q in (-1,0,1):
        H_hfs += b_hfs * delta(A.N,B.N) * delta(A.mN,B.mN) * delta(A.mF(),B.mF()) * (-1)**q * (-1)**(A.S-A.mS) * wigner_3j(A.S,1,B.S,-A.mS,q,B.mS) * (-1)**(A.I-A.mI) * wigner_3j(A.I,1,B.I,-A.mI,-q,B.mI) * np.sqrt(A.S*(A.S+1)*(2*A.S+1)) * np.sqrt(A.I*(A.I+1)*(2*A.I+1))
        H_hfs += c_hfs * ((delta(A.N,B.N) * delta(A.mN,B.mN) * delta(A.mS,B.mS) * delta(A.mI,B.mI) * (A.mS*A.mI)))
        H_hfs += -c_hfs * ((1/3) * delta(A.N,B.N) * delta(A.mN,B.mN) * delta(A.mF(),B.mF()) * (-1)**q * (-1)**(A.S-A.mS) * wigner_3j(A.S,1,B.S,-A.mS,q,B.mS) * (-1)**(A.I-A.mI) * wigner_3j(A.I,1,B.I,-A.mI,-q,B.mI) * np.sqrt(A.S*(A.S+1)*(2*A.S+1)) * np.sqrt(A.I*(A.I+1)*(2*A.I+1)))
    '''
    The two c_hfs terms come about due to the non-spherical, but present cylindrical symmetry
    of the diatomic representation of YbOh. If we choose to ignore them, then the energy levels
    will behave well as if it were atomic energy levels
    '''
    return H_hfs

def H_mag(A,B):
    # external B-field S.B
    H_magnetic = delta(A.mI,B.mI)*delta(A.mS,B.mS)*delta(A.mN,B.mN)*delta(A.N,B.N) * (gS*A.mS + gI*A.mI)
    H_magnetic = -muB * H_magnetic
    return H_magnetic

## Basis set and calculations
ground_basis = [MolecularState(N=i,I=1/2,S=1/2) for i in range(2)] # range(x) goes from 0,1,2,...,(x-1) 
ground_basis = sublevel_expand(ground_basis) # expande the basis to contain all the mN,mI,mS sublevels

num_i = 0
num_f = len(ground_basis)
N = (num_f - num_i)

## magnetic shift plot
H0 = np.matrix(np.zeros((N,N))) # Create N x N zero matrix
for i in range(num_i,num_f):
    for j in range(num_i,num_f):
        A,B = ground_basis[j], ground_basis[i]
        H0[j,i] = H_rot(A,B)
        H0[j,i] += H_sr(A,B)
        H0[j,i] += H_hfs(A,B)
        
H_int = np.matrix(np.zeros((N,N)))
for i in range(num_i,num_f):
    for j in range(num_i,num_f):
        A,B = ground_basis[j], ground_basis[i]
        H_int[j,i] = H_mag(A,B)   
        
#%%
class ExcitedMolecularState():
    def __init__(self,J=1/2,I=1/2):
        self.J=J
        self.mJ = J
        self.I=I
        self.mI = I
        
    def __repr__(self):
        attribs = [str(s) for s in [self.J,self.mJ,self.I,self.mI]]
        string = ','.join([str(s) for s in attribs])
        return "|" + string + "; " + "mF=" + str(self.excited_mF()) + ">"
    
    def excited_mF(self):
        return self.mJ + self.mI
    
    def excited_F(self):
        return self.J + self.I

def excited_sublevel_expand(basis):
    newbasis = []
    for ket in basis:
        for mJ in np.arange(-ket.J,ket.J+1,1):
            for mI in np.arange(-ket.I,ket.I+1,1):
                newket = deepcopy(ket)
                newket.mJ = mJ
                newket.mI = mI
                newbasis.append(newket)
    return newbasis

def delta(i,j): 
    if i==j: return 1
    else: return 0
    
## molecule properties & functions
muB = 14        # GHz/T
gL = -1
gS = -2.0023
gJ = -0.2#-0.002 # Unsure exactly what order of magnitude gJ should come out to be, play with this parameter which also scales the hyperfine constants proportionally
gI = 5.585 * 1/1836.152672   # for H nucleus; using same muB for both terms, we divide this term by the difference in magnitude between muB and muN

wavenum_freq = 100*(3*10**8)*1e-9 # convert from cm^-1 to GHz
B_rot_excited = B_rot = 7348.4005e-3     # GHz ; molecular rotation constant
b_hfs_excited = 4.80e-3*gJ  # GHz ; one of the hyperfine constants
c_hfs_excited = 2.46e-3*gJ  # GHz ; another hyperfine constant

# EXCITED() STATE HAMILTONIANS
def H_rot_excited(A,B):
    # rotational J(J+1)
    H_rot_excited =  delta(A.mI,B.mI) * delta(A.J,B.J) * delta(A.mJ,B.mJ) * (B_rot * A.J * (A.J+1))
    return H_rot_excited

def H_hfs_excited(A,B):
    # hfs I.J
    H_hfs_excited = 0
    for q in (-1,0,1):
        H_hfs_excited += b_hfs_excited * delta(A.J,B.J) * delta(A.excited_mF(),B.excited_mF()) * (-1)**q * (-1)**(A.J-A.mJ) * wigner_3j(A.J,1,B.J,-A.mJ,q,B.mJ) * (-1)**(A.I-A.mI) * wigner_3j(A.I,1,B.I,-A.mI,-q,B.mI) * np.sqrt(A.J*(A.J+1)*(2*A.J+1)) * np.sqrt(A.I*(A.I+1)*(2*A.I+1))
        H_hfs_excited += c_hfs_excited * delta(A.J,B.J) * delta(A.mJ,B.mJ) * delta(A.mI,B.mI) * (A.mJ*A.mI)
        H_hfs_excited += -c_hfs_excited * (1/3) * delta(A.J,B.J) * delta(A.excited_mF(),B.excited_mF()) * (-1)**q * (-1)**(A.J-A.mJ) * wigner_3j(A.J,1,B.J,-A.mJ,q,B.mJ) * (-1)**(A.I-A.mI) * wigner_3j(A.I,1,B.I,-A.mI,-q,B.mI) * np.sqrt(A.J*(A.J+1)*(2*A.J+1)) * np.sqrt(A.I*(A.I+1)*(2*A.I+1))
    return H_hfs_excited

def H_mag_excited(A,B):
    # external B-field S.B
    H_magnetic_excited = delta(A.mI,B.mI) * delta(A.J,B.J) * delta(A.mJ,B.mJ) * (gJ*A.mJ + gI*A.mI)
    H_magnetic_excited = -muB * H_magnetic_excited
    return H_magnetic_excited

## Basis set and calculations
excited_basis = [ExcitedMolecularState(J=(1/2+i),I=1/2) for i in range(2)] # range(x) goes from 0,1,2,...,(x-1) 
excited_basis = excited_sublevel_expand(excited_basis) # expande the basis to contain all the mN,mI,mS sublevels
N_excited = len(excited_basis)

## magnetic shift plot
H0_excited = np.matrix(np.zeros((N_excited,N_excited))) # Create N x N zero matrix

for i in range(N_excited):
    for j in range(N_excited):
        A,B = excited_basis[j], excited_basis[i]
        H0_excited[j,i] = H_rot_excited(A,B)
        H0_excited[j,i] += H_hfs_excited(A,B)
        
H_int_excited = np.matrix(np.zeros((N_excited,N_excited)))
for i in range(N_excited):
    for j in range(N_excited):
        A,B = excited_basis[j], excited_basis[i]
        H_int_excited[j,i] = H_mag_excited(A,B)   
#%% Print eigenvalues and eigenvectors
        
B_scale = 2.15 # Typical field we would produce is on the order of 2.5T
B_max = B_scale*fullfield(0,0,10e-3,stagescale)   

z_position = 10e-3 #in units of mm
B_val = B_scale*fullfield(0,0,z_position,stagescale)
Bfield = np.linspace(B_val,B_val,1)

z_position2 = 5.4e-3 #in units of mm
B_val2 = B_scale*fullfield(0,0,z_position2,stagescale)
Bfield2 = np.linspace(B_val2,B_val2,1) 

ground_energies, ground_eigenvectors = [lg.eigh(H0 + Bfield[i]*H_int)  for i in range(len(Bfield))][0]
excited_energies, excited_eigenvectors = [lg.eigh(H0_excited + Bfield[i]*H_int_excited)  for i in range(len(Bfield))][0]


ground_energies2, ground_eigenvectors2 = [lg.eigh(H0 + Bfield2[i]*H_int)  for i in range(len(Bfield2))][0]
excited_energies2, excited_eigenvectors2 = [lg.eigh(H0_excited + Bfield2[i]*H_int_excited)  for i in range(len(Bfield2))][0]

for i in range(4):
    print("Energy = {}".format(round(ground_energies[i],4)))
    print(ground_eigenvectors.H[i],"\n")
    
for i in range(4):
    print("Energy = {}".format(round(excited_energies[i],10)))
    print(excited_eigenvectors.H[i],"\n") 

#%% YbOH PLOTS

# PLOT OF THE ANALYTIC MAGNETIC FIELD
plt.figure()
plt.grid(True)
plt.plot(z*1e3, B_scale*fullfield(0,0,z,stagescale))
plt.title("Analytic magnetic field from two stages of a Halbach cylinder")
plt.xlabel("Position [mm]")
plt.ylabel("Magnetic field [T]")
plt.margins(0)
plt.show()

# ENERGY vs MAGNETIC FIELD
eta = np.linspace(0,B_max,1e4)        # T
plt.figure()
plt.grid(True)
plt.plot(eta, [lg.eigh(H0 + eta[i]*H_int)[0] for i in range(len(eta))]) # for every eigenvalue, plot H0 + Hint = H0 + muB*B(gs + gI)
plt.title("$\~{X}^2\Sigma$ state spectrum for YbOH in a magnetic field")
plt.xlabel("B-field [T]")
plt.ylabel("Energy [GHz]")
plt.margins(0)
plt.show()

plt.figure()
plt.grid(True)
plt.plot(eta, [lg.eigh(H0_excited + eta[i]*H_int_excited)[0] for i in range(len(eta))]) # for every eigenvalue, plot H0 + Hint = H0 + muB*B(gs + gI)
plt.title("$\~{A}^2\Pi_{1/2}$ state spectrum for YbOH in a magnetic field")
plt.xlabel("B-field [T]")
plt.ylabel("Energy [GHz]")
plt.margins(0)
plt.show()

#ENERGY vs ANALYTIC MAGNETIC FIELD (HALBACH CYLINDER)
plt.figure()
plt.grid(True)
plt.plot(z*1e3, [lg.eigh(H0 + B_scale*fullfield(0,0,z,stagescale)[i]*H_int)[0]  for i in range(len(z))])
plt.title("$\~{X}^2\Sigma$ state spectrum for YbOH in the Halbach field")
plt.xlabel("Position [mm]")
plt.ylabel("Energy [GHz]")
plt.margins(0)
plt.show()

plt.figure()
plt.grid(True)
plt.plot(z*1e3, [lg.eigh(H0_excited + B_scale*fullfield(0,0,z,stagescale)[i]*H_int_excited)[0]  for i in range(len(z))])
plt.title("$\~{A}^2\Pi_{1/2}$ state spectrum for YbOH in the Halbach field")
plt.xlabel("Position [mm]")
plt.ylabel("Energy [GHz]")
plt.margins(0)
plt.show()

#%%
import matplotlib.lines as mlines
red_line = mlines.Line2D([], [], color='red',markersize=15, label='$\~{A}^2\Pi_{1/2}$ state')
blue_line = mlines.Line2D([], [], color='blue',markersize=15, label='$\~{X}^2\Sigma$ state')
'''
# PLOT OF THE ANALYTIC MAGNETIC FIELD
plt.figure()
plt.grid(True)
plt.plot(z*1e3, B_scale*fullfield(0,0,z,stagescale))
plt.title("Analytic magnetic field produced by two stages of a Halbach cylinder")
plt.xlabel("Position [mm]")
plt.ylabel("Magnetic field [T]")
plt.margins(0)
plt.show()

eta = np.linspace(0,B_scale*fullfield(0,0,10e-3,stagescale),1e4)        # T
plt.figure()
plt.grid(True)
plt.plot(eta, [lg.eigh(H0 + eta[i]*H_int)[0] for i in range(len(eta))],color = "blue") # for every eigenvalue, plot H0 + Hint = H0 + muB*B(gs + gI)
plt.plot(eta, [lg.eigh(H0_excited + eta[i]*H_int_excited)[0] for i in range(len(eta))],color = "red") # for every eigenvalue, plot H0 + Hint = H0 + muB*B(gs + gI)
plt.legend(handles=[red_line,blue_line])
plt.title("$\~{X}^2\Sigma$ and $\~{A}^2\Pi_{1/2}$ state spectra for YbOH in a magnetic field")
plt.xlabel("B-field [T]")
plt.ylabel("Energy [GHz]")
plt.margins(0)
plt.show()
'''
plt.figure()
plt.grid(True)
plt.plot(z*1e3, [lg.eigh(H0 + B_scale*fullfield(0,0,z,stagescale)[i]*H_int)[0]  for i in range(len(z))],color = "blue")
plt.plot(z*1e3, [lg.eigh(H0_excited + B_scale*fullfield(0,0,z,stagescale)[i]*H_int_excited)[0]  for i in range(len(z))],color = "red")
plt.legend(handles=[red_line,blue_line])
plt.arrow(z_position/mm, ground_energies[1], 0, excited_energies[-2]-ground_energies[1], head_width=0.30, head_length=1.55, width = 0.06, length_includes_head = True)
plt.arrow(z_position2/mm, ground_energies2[1], 0, excited_energies2[-2]-ground_energies2[1], head_width=0.30, head_length=1.55, width = 0.06, length_includes_head = True)

t = ("???")
plt.text(11, -5, t, fontsize=18, style='oblique', ha='center',rotation=15,va='top', wrap=True)

t1 = (r"$\vert \alpha \rangle$")
plt.text(10.2, -31.5, t1, fontsize=14, wrap=True)

t2 = (r"$\vert \beta \rangle$")
plt.text(10.2, 40, t2, fontsize=14, wrap=True)

plt.title("$\~{X}^2\Sigma$ and $\~{A}^2\Pi_{1/2}$ state spectra for YbOH in the Halbach field")
plt.xlabel("Position [mm]")
plt.ylabel("Energy [GHz]")
plt.margins(0.02)
plt.show()
#%%
## dipole matrix element between ground and excited states

def dipole_matrix_element(A,B,q=0):
    # assumption: A is an excited basis state
    #             B is a ground basis state
    # this returns <e_i|D_q|g_j>
    dipole_matrix_element = 0
#    for Q in (-1,0,1):
#        dipole_matrix_element += delta(A.mI,B.mI)* (-1)**(B.J()+1/2+0) * np.sqrt(2*B.N+1) * wigner_3j(B.S,B.N,B.J(),1/2,0,-1/2) * (-1)**(A.J-A.mJ) * wigner_3j(A.J,1,B.J(),-A.mJ,q,B.mJ()) * (-1)**(A.J - 1/2) * wigner_3j(A.J,1,B.J(),-1/2,Q,1/2) # * < Omega'|| D || Omega >
#    return dipole_matrix_element
    for Q in (-1,0,1):
        for Sigma in (-1/2,1/2):
            for Omega in (-1/2,1/2):
                for Omega_e in (-1/2,1/2):
                    dipole_matrix_element += delta(A.mI,B.mI)* (-1)**(B.J()+Sigma+0) * np.sqrt(2*B.N+1) * wigner_3j(B.S,B.N,B.J(),Sigma,0,-Omega) * (-1)**(A.J-A.mJ) * wigner_3j(A.J,1,B.J(),-A.mJ,q,B.mJ()) * (-1)**(A.J - Omega_e) * wigner_3j(A.J,1,B.J(),-Omega_e,Q,Omega) # * < Omega'|| D || Omega >
    return dipole_matrix_element
     
def dipole_matrix(excited_basis, ground_basis, q=0):
    N_excited = len(excited_basis)
    N_ground = len(ground_basis)
    E1_matrix_q = np.matrix(np.zeros((N_excited,N_ground))) 
    for j in range(N_excited):
        for i in range(N_ground):
            A,B = excited_basis[j], ground_basis[i]
            E1_matrix_q[j,i] = dipole_matrix_element(A,B,q)
    return E1_matrix_q

D_0 = dipole_matrix(excited_basis,ground_basis,q=0)
D_plus = dipole_matrix(excited_basis,ground_basis,q=+1)
D_minus = dipole_matrix(excited_basis,ground_basis,q=-1)

def transition_amplitude(v_A,v_B,q=0,E_field_amplitude=1):
    # assumption: v_A is any excited state (linear combo of excited basis states),
    #             v_B is any ground state (linear combo of ground basis states)    
    if q==0: 
        return (-1)**q * v_A.H * D_0 * v_B * E_field_amplitude        
    elif q==-1:
        return (-1)**q * v_A.H * D_minus * v_B * E_field_amplitude
    elif q==+1:
        return (-1)**q * v_A.H * D_plus * v_B * E_field_amplitude
#%%
#transition_amplitude(excited_eigenvectors.H,ground_eigenvectors.H,q=0)[8]
transition_amplitude(excited_eigenvectors2.H,ground_eigenvectors2.H,q=0)[8]

#Tot_transition_amp = np.add(np.add(transition_amplitude(excited_eigenvectors.H,ground_eigenvectors.H,q=1),transition_amplitude(excited_eigenvectors.H,ground_eigenvectors.H,q=-1)),transition_amplitude(excited_eigenvectors.H,ground_eigenvectors.H,q=0))
#Tot_transition_amp[0]
#%%
print(D_0,'\n')
print(D_plus,'\n')
print(D_minus)

#%%
'''
## dipole matrix element between excited and ground states

def dipole_matrix_element_2(B,A,q=0):
    # assumption: A is an excited basis state
    #             B is a ground basis state
    # this returns <e_i|D_q|g_j>
    dipole_matrix_element_2 = 0
    for qN in ((A.J-1/2),(A.J+1/2)):
        for Sigma in (-1/2,1/2):
            for Lambda in (-1,0,1):
                dipole_matrix_element_2 += delta(A.mJ,B.mJ())*delta(A.J,B.J())*delta((A.J-1/2),B.N)*delta(A.mI,B.mI)* (-1)**(A.J + Sigma + Lambda) * np.sqrt(2*qN+1) * wigner_3j(1/2,qN,A.J,Sigma,Lambda,-1/2)
    return dipole_matrix_element_2
    
def dipole_matrix_2(ground_basis, excited_basis, q=0):
    N_excited = len(excited_basis)
    N_ground = len(ground_basis)
    E2_matrix_q = np.matrix(np.zeros((N_ground,N_excited))) 
    for j in range(N_excited):
        for i in range(N_ground):
            A,B = excited_basis[j], ground_basis[i]
            E2_matrix_q[i,j] = dipole_matrix_element_2(B,A,q)
    return E2_matrix_q

D_0_2 = dipole_matrix_2(ground_basis,excited_basis,q=0)
D_plus_2 = dipole_matrix_2(ground_basis,excited_basis,q=+1)
D_minus_2 = dipole_matrix_2(ground_basis,excited_basis,q=-1)

def branching_ratio(v_B,v_A,q=0,E_field_amplitude=1):
    # assumption: v_A is any excited state (linear combo of excited basis states),
    #             v_B is any ground state (linear combo of ground basis states)    
    if q==0: 
        return (-1)**q * v_B.H * D_0_2 * v_A * E_field_amplitude        
    elif q==-1:
        return (-1)**q * v_B.H * D_minus_2 * v_A * E_field_amplitude
    elif q==+1:
        return (-1)**q * v_B.H * D_plus_2 * v_A * E_field_amplitude
    
#%%
#print(excited_eigenvectors.H,"\n")
#print(ground_eigenvectors.H)
branching_ratio(ground_eigenvectors.H,excited_eigenvectors.H,q=0)

#%%
print(D_0_2,'\n')
print(D_plus_2,'\n')
print(D_minus_2)
'''